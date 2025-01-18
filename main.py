"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

from dataclasses import dataclass
import time
import traceback
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf
from viser import ViserServer, GuiApi
from viser._gui_handles import GuiMarkdownHandle, GuiSliderHandle
from viser._scene_api import SceneApi
from viser._scene_handles import GaussianSplatHandle

from llm import prompt_llm
import prompts


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }


def load_splat(path: Path) -> SplatFile:
    if path.suffix == ".splat":
        return load_splat_file(path, center=True)
    elif path.suffix == ".ply":
        return load_ply_file(path, center=True)
    else:
        raise SystemExit("Please provide a filepath to a .splat or .ply file.")

class Observer:
    def update(self):
        pass

class Subject:
    def __init__(self):
        self._observers: list[Observer] = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update()

class State(Subject):
    def __init__(self, splat_path: Path, fps: int = 24):
        super().__init__()
        self.splat_path = splat_path
        self.fps = fps

        self.splat: SplatFile = load_splat(splat_path)
        self._animation_name: str = ""
        self.animation_prompt: str = ""
        self.temperature: float = 0.7
        self.centers_fn_md: str = "```\n```"
        self.rgbs_fn_md: str = "```\n```"
        self.opacities_fn_md: str = "```\n```"
        self.frame_to_gs_handle: dict[int, GaussianSplatHandle] = {}

    @property
    def animation_name(self):
        return self._animation_name

    @animation_name.setter
    def animation_name(self, value: str):
        self._animation_name = value
        self.notify()



class Gui(Observer):
    def __init__(self, server: ViserServer, state: State):
        self.api: GuiApi = server.gui
        self.state: State = state

        self.api.configure_theme(dark_mode=True)

        self.animation_md = self.api.add_markdown("Animation: **None**")

        self.prompt_btn = self.api.add_button("Prompt", icon=viser.Icon.ROBOT)
        @self.prompt_btn.on_click
        def _(_) -> None:
            open_prompt(scene=server.scene, gui=self, state=self.state)

        self.inspect_btn = self.api.add_button("Inspect Functions", icon=viser.Icon.FUNCTION)
        @self.inspect_btn.on_click
        def _(_) -> None:
            open_inspector(gui=self, state=self.state)

        self.reload_btn = server.gui.add_button("Reload Frames", icon=viser.Icon.RESTORE)
        @self.reload_btn.on_click
        def _(_) -> None:
            remove_gs_handles(self.state)
            load_splats(server.scene, self, self.state)
            change_to_frame(self.frame_slider.value, self.state)

        self.speed_dropdown = self.api.add_dropdown(
            label="Speed",
            options=["0.25x", "0.5x", "1x", "2x"],
            initial_value="1x"
        )

        self.frame_slider = self.api.add_slider(
            label=f"Frame ({self.state.fps}fps)",
            min=0,
            max=self.state.fps-1,
            step=1,
            initial_value=0,
        )
        @self.frame_slider.on_update
        def _(_) -> None:
            change_to_frame(frame=self.frame_slider.value, state=self.state)

        self.play_btn = server.gui.add_button("▶ Play", color="green")
        @self.play_btn.on_click
        def _(_) -> None:
            self.play_btn.disabled = True
            play_animation(gui=self, state=self.state)
            self.play_btn.disabled = False

    def update(self):
        self.animation_md.content = f"Animation: **{self.state.animation_name}**"

def extract_python_code(markdown: str):
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    match = pattern.search(markdown)
    return match.group(1) if match else ""

def generate_functions(prompt: str, temperature: float, gui: Gui):
    status = gui.api.add_markdown("*Analyzing Prompt...*")
    progress = gui.api.add_progress_bar(value=0, animated=True)

    centers_summary = prompt_llm(prompt=prompt, system_message=prompts.CENTERS_SUMMARY, temperature=temperature)
    progress.value = 10
    rgbs_summary = prompt_llm(prompt=prompt, system_message=prompts.RGBS_SUMMARY, temperature=temperature)
    progress.value = 20
    opacities_summary = prompt_llm(prompt=prompt, system_message=prompts.OPACITIES_SUMMARY, temperature=temperature)
    progress.value = 30

    status.content = "*Generating Functions...*"
    raw_centers_function = prompt_llm(prompt=centers_summary, system_message=prompts.CENTERS_GENERATOR, temperature=temperature)
    progress.value = 40
    raw_rgbs_function = prompt_llm(prompt=rgbs_summary, system_message=prompts.RGBS_GENERATOR, temperature=temperature)
    progress.value = 50
    raw_opacities_function = prompt_llm(prompt=opacities_summary, system_message=prompts.OPACITIES_GENERATOR, temperature=temperature)
    progress.value = 60

    status.content = "*Validating Python...*"
    validated_centers_function = prompt_llm(prompt=raw_centers_function, system_message=prompts.PYTHON_VALIDATION, temperature=temperature)
    progress.value = 70
    validated_rgbs_function = prompt_llm(prompt=raw_rgbs_function, system_message=prompts.PYTHON_VALIDATION, temperature=temperature)
    progress.value = 80
    validated_opacities_function = prompt_llm(prompt=raw_opacities_function, system_message=prompts.PYTHON_VALIDATION, temperature=temperature)
    progress.value = 90

    executable_centers_function = extract_python_code(validated_centers_function)
    executable_rgbs_function = extract_python_code(validated_rgbs_function)
    executable_opacities_function = extract_python_code(validated_opacities_function)
    
    exec(executable_centers_function, globals())
    exec(executable_rgbs_function, globals())
    exec(executable_opacities_function, globals())

    progress.value = 100
    progress.remove()
    status.remove()

    return executable_centers_function, executable_rgbs_function, executable_opacities_function


def check_functions_defined() -> bool:
    fn_names = ["compute_centers", "compute_rgbs", "compute_opacities"]
    return all(name in globals() for name in fn_names)

def open_debug(error_msg: str, t: float, gui: Gui, state: State) -> None:
    with gui.api.add_modal("🐞 Debugger") as popout:
        gui.api.add_markdown(
            f"At t=**{np.round(t, 3)}** the following error occurred:\n```{error_msg}\n```")
        
        close_btn = gui.api.add_button("X", color="red")
        @close_btn.on_click
        def _(_) -> None:
            popout.close()

def add_splat_at_t(t: float, scene: SceneApi, gui: Gui, state: State):
    if not check_functions_defined():
        with gui.api.add_modal("⚠ Please generate functions first!") as popout:
            close_btn = gui.api.add_button("X", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()
            raise Exception()

    if np.isclose(t, 0.0):
        return scene.add_gaussian_splats(
        f"splat_0",
        centers=state.splat["centers"],
        rgbs=state.splat["rgbs"],
        opacities=state.splat["opacities"],
        covariances=state.splat["covariances"],
        )
    
    try:
        centers = state.splat["centers"]
        rgbs = state.splat["rgbs"]
        opacities = state.splat["opacities"]
        centers_at_t = globals()["compute_centers"](t, centers)
        rbgs_at_t = globals()["compute_rgbs"](t, rgbs)
        opacities_at_t = globals()["compute_opacities"](t, opacities)
        return scene.add_gaussian_splats(
            f"splat_{t}",
            centers=centers_at_t,
            rgbs=rbgs_at_t,
            opacities=opacities_at_t,
            covariances=state.splat["covariances"],
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        open_debug(error_msg=stack_trace, t=t, gui=gui, state=state)
        raise

def load_splats(scene: SceneApi, gui: Gui, state: State) -> None:
    loading_md = gui.api.add_markdown("*Loading Frames...*")
    progress_bar = gui.api.add_progress_bar(0.0, animated=True)
    seconds_per_frame = 1.0 / state.fps
    for frame in range(state.fps):
        t = frame * seconds_per_frame
        try:
            gs_handle = add_splat_at_t(t, scene, gui, state)
            gs_handle.visible = False
            state.frame_to_gs_handle[frame] = gs_handle
            progress_bar.value = ((frame+1)/state.fps) * 100
        except Exception:
            break
    change_to_frame(gui.frame_slider.value, state)
    progress_bar.remove()
    loading_md.remove()

def as_code_block(value: str) -> str:
    return f"```\n{value}\n```"

def open_prompt(scene: SceneApi, gui: Gui, state: State):
    with gui.api.add_modal(title="🤖 Animation Generator") as popout:
        name_txt = gui.api.add_text("Name", state.animation_name)
        input_txt = gui.api.add_text("Prompt", state.animation_prompt)
        temperatur_slider = gui.api.add_slider(
            label="Temperature",
            min=0.0,
            max=1.0,
            step=0.1,
            initial_value=state.temperature
        )

        generate_btn = gui.api.add_button("🛠 Generate")
        @generate_btn.on_click
        def _(_) -> None:
            if not input_txt.value:
                return
            state.animation_name = name_txt.value
            state.animation_prompt = input_txt.value
            state.temperature = temperatur_slider.value
            popout.close()
            centers_fn_code, rgbs_fn_code, opacities_fn_code = generate_functions(prompt=input_txt.value, temperature=temperatur_slider.value, gui=gui)
            state.centers_fn_md = as_code_block(centers_fn_code)
            state.rgbs_fn_md = as_code_block(rgbs_fn_code)
            state.opacities_fn_md = as_code_block(opacities_fn_code)
            remove_gs_handles(state)
            load_splats(scene, gui, state)

        abort_btn = gui.api.add_button("Abort", icon=viser.Icon.X, color="red")
        @abort_btn.on_click
        def _(_) -> None:
            popout.close()

def open_inspector(gui: Gui, state: State):
    with gui.api.add_modal("🔎 Function Inspector") as popout:
        gui.api.add_markdown(state.centers_fn_md)
        gui.api.add_markdown(state.rgbs_fn_md)
        gui.api.add_markdown(state.opacities_fn_md)

        close_btn = gui.api.add_button("X", color="red")
        @close_btn.on_click
        def _(_) -> None:
            popout.close()

def change_to_frame(frame: int, state: State):
    for i, gs_handle in state.frame_to_gs_handle.items():
        if i == frame:
            gs_handle.visible = True
        else:
            gs_handle.visible = False

def speed_to_sleep_factor(speed: str) -> float:
    if speed == "0.25x":
        return 4.0
    elif speed == "0.5x":
        return 2.0
    elif speed == "1x":
        return 1.0
    elif speed == "2x":
        return 0.5
    else:
        return 1.0

def play_animation(gui: Gui, state: State):
    animation_stopped = False

    stop_btn = gui.api.add_button("◼ Stop", color="red")
    @stop_btn.on_click
    def _(_) -> None:
        nonlocal animation_stopped
        animation_stopped=True
        stop_btn.remove()

    while True:
        for frame in range(gui.frame_slider.value+1, state.fps):
            if animation_stopped:
                return
            change_to_frame(frame, state)
            gui.frame_slider.value = frame
            time.sleep((1.0/state.fps) * speed_to_sleep_factor(gui.speed_dropdown.value))
        change_to_frame(0, state)
        gui.frame_slider.value = 0

def remove_gs_handles(state: State) -> None:
    for _, gs_handle in state.frame_to_gs_handle.items():
        if gs_handle:
            gs_handle.remove()
            del gs_handle

def main(splat_path: Path) -> None:
    server = viser.ViserServer()
    server.scene.add_transform_controls("0",scale=0.3, opacity=0.3)

    state = State(splat_path, fps=24)
    gui = Gui(server, state)
    state.attach(gui)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        remove_gs_handles(state)
        raise


if __name__ == "__main__":
    tyro.cli(main)
