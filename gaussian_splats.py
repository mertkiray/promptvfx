"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
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

def extract_python_code(markdown: str):
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    match = pattern.search(markdown)
    return match.group(1) if match else ""

def generate_functions(prompt: str, gui: GuiApi):
    status = gui.add_markdown("*Analyzing Prompt...*")
    progress = gui.add_progress_bar(value=0, animated=True)

    centers_summary = prompt_llm(prompt=prompt, system_message=prompts.CENTERS_SUMMARY)
    progress.value = 10
    rgbs_summary = prompt_llm(prompt=prompt, system_message=prompts.RGBS_SUMMARY)
    progress.value = 20
    opacities_summary = prompt_llm(prompt=prompt, system_message=prompts.OPACITIES_SUMMARY)
    progress.value = 30

    status.content = "*Generating Functions...*"
    raw_centers_function = prompt_llm(prompt=centers_summary, system_message=prompts.CENTERS_GENERATOR)
    progress.value = 40
    raw_rgbs_function = prompt_llm(prompt=rgbs_summary, system_message=prompts.RGBS_GENERATOR)
    progress.value = 50
    raw_opacities_function = prompt_llm(prompt=opacities_summary, system_message=prompts.OPACITIES_GENERATOR)
    progress.value = 60

    status.content = "*Validating Python...*"
    validated_centers_function = prompt_llm(prompt=raw_centers_function, system_message=prompts.PYTHON_VALIDATION)
    progress.value = 70
    validated_rgbs_function = prompt_llm(prompt=raw_rgbs_function, system_message=prompts.PYTHON_VALIDATION)
    progress.value = 80
    validated_opacities_function = prompt_llm(prompt=raw_opacities_function, system_message=prompts.PYTHON_VALIDATION)
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

def add_splat_at_t(t: float, server: viser.ViserServer, splat_data: SplatFile):
    all_functions_defined = all(key in globals() for key in ["update_centers", "update_rgbs", "update_opacities"])

    if np.isclose(t, 0.0) or not all_functions_defined:
        return server.scene.add_gaussian_splats(
        f"splat_{t}",
        centers=splat_data["centers"],
        rgbs=splat_data["rgbs"],
        opacities=splat_data["opacities"],
        covariances=splat_data["covariances"],
        )
    
    updated_centers = globals()["update_centers"](splat_data["centers"], t)
    updated_rgbs = globals()["update_rgbs"](splat_data["rgbs"], t)
    updated_opacities = globals()["update_opacities"](splat_data["opacities"], t)

    return server.scene.add_gaussian_splats(
        f"splat_{t}",
        centers=updated_centers,
        rgbs=updated_rgbs,
        opacities=updated_opacities,
        covariances=splat_data["covariances"],
    )

def load_splats(frames: int, splat_data: SplatFile, server: ViserServer) -> dict[int, GaussianSplatHandle]:
    status = server.gui.add_markdown("*Loading Frames...*")
    progress = server.gui.add_progress_bar(0.0, animated=True)
    seconds_per_frame = 1.0 / frames
    frame_to_handle: dict[int, GaussianSplatHandle] = {}
    for frame in range(frames):
        t = frame * seconds_per_frame
        gs_handle = add_splat_at_t(
            t,
            server,
            splat_data.copy(),)
        gs_handle.visible = False
        frame_to_handle[frame] = gs_handle
        progress.value = (frame+1)/frames
    progress.remove()
    status.remove()
    return frame_to_handle

def open_prompt(gui: GuiApi, animation_name_handle: GuiMarkdownHandle, functions_to_md: dict[str, str]):
    with gui.add_modal(title="Animation Generator 🤖") as popout:
        name_field = gui.add_text("Name", "New Animation")
        input_field = gui.add_text("Prompt", "")

        generate_button = gui.add_button("🛠 Generate")
        @generate_button.on_click
        def _(_) -> None:
            if not input_field.value:
                return
            animation_name_handle.content = f"Animation: **{name_field.value}**"
            popout.close()
            centers_md, rgbs_md, opacities_md = generate_functions(prompt=input_field.value, gui=gui)
            functions_to_md["centers"] = f"```\n{centers_md}\n```"
            functions_to_md["rgbs"] = f"```\n{rgbs_md}\n```"
            functions_to_md["opacities"] = f"```\n{opacities_md}\n```"

        abort_button = gui.add_button("Abort", icon=viser.Icon.X, color="red")
        @abort_button.on_click
        def _(_) -> None:
            popout.close()


def open_function_inspector(gui: GuiApi, centers_md: str, rgbs_md: str, opacities_md: str):
    with gui.add_modal("🔎 Function Inspector") as popout:
        gui.add_markdown(centers_md)
        gui.add_markdown(rgbs_md)
        gui.add_markdown(opacities_md)
        close = gui.add_button("X", color="red")
        @close.on_click
        def _(_) -> None:
            popout.close()

def change_to_frame(frame: int, frame_to_handle: dict[int, GaussianSplatHandle]):
    for i, handle in frame_to_handle.items():
        if i == frame:
            handle.visible = True
        else:
            handle.visible = False

def play_animation(gui: GuiApi, frame_slider: GuiSliderHandle, frame_to_handle: dict[int, GaussianSplatHandle]):
    animation_stopped = False
    current_frame = frame_slider.value
    total_frames = int(frame_slider.max) + 1

    gui_stop_button = gui.add_button("Stop", color="red", order=5)
    @gui_stop_button.on_click
    def _(_) -> None:
        nonlocal animation_stopped
        animation_stopped=True
        gui_stop_button.remove()

    for frame in range(current_frame, total_frames):
        if animation_stopped:
            return
        change_to_frame(frame, frame_to_handle)
        frame_slider.value = frame
        time.sleep((1.0/total_frames)*2)
    
    change_to_frame(0, frame_to_handle)
    frame_slider.value = 0
    gui_stop_button.remove()


def remove_handles(frame_to_handle: dict[int, GaussianSplatHandle]) -> None:
    for _, handle in frame_to_handle.items():
        if handle:
            handle.remove()
            del handle

def main(splat_path: Path) -> None:
    server = viser.ViserServer()
    server.scene.add_transform_controls("0",scale=0.3, opacity=0.3)
    server.gui.configure_theme(dark_mode=True)

    if splat_path.suffix == ".splat":
        splat_data = load_splat_file(splat_path, center=True)
    elif splat_path.suffix == ".ply":
        splat_data = load_ply_file(splat_path, center=True)
    else:
        raise SystemExit("Please provide a filepath to a .splat or .ply file.")

    # Animation Settings
    FRAMES = 12

    # State
    functions_to_md: dict[str, str] = {
        "centers": "```\n```",
        "rgbs": "```\n```",
        "opacities": "```\n```",
    }
    frame_to_handle: dict[int, GaussianSplatHandle] = {}

    gui_animation_name = server.gui.add_markdown("Animation: **None**")

    gui_prompt_button = server.gui.add_button(label="Prompt", icon=viser.Icon.ROBOT, order=1)
    @gui_prompt_button.on_click
    def _(_) -> None:
        open_prompt(
            gui=server.gui,
            animation_name_handle=gui_animation_name,
            functions_to_md=functions_to_md)
    

    gui_inspect_button = server.gui.add_button(
        "🔎 Inspect Functions",
        order=2
    )
    @gui_inspect_button.on_click
    def _(_) -> None:
        open_function_inspector(
            gui=server.gui,
            centers_md=functions_to_md["centers"],
            rgbs_md=functions_to_md["rgbs"],
            opacities_md=functions_to_md["opacities"]
        )

    gui_frame_slider = server.gui.add_slider(
        label=f"Frame ({FRAMES}fps)",
        min=0,
        max=FRAMES-1,
        step=1,
        initial_value=0,
        order=4
    )
    @gui_frame_slider.on_update
    def _(_) -> None:
        change_to_frame(gui_frame_slider.value, frame_to_handle)

    gui_play_button = server.gui.add_button(
        label="Play",
        color="green",
        icon=viser.Icon.BRAND_GOOGLE_PLAY,
        order=4
    )
    @gui_play_button.on_click
    def _(_) -> None:
        gui_play_button.disabled = True
        play_animation(
            gui=server.gui,
            frame_slider=gui_frame_slider,
            frame_to_handle=frame_to_handle
        )
        gui_play_button.disabled = False


    gui_reload_button = server.gui.add_button("Reload Splats", icon=viser.Icon.RESTORE, order=3)
    @gui_reload_button.on_click
    def _(_) -> None:
        nonlocal frame_to_handle
        if frame_to_handle:
            remove_handles(frame_to_handle)
        frame_to_handle = load_splats(FRAMES, splat_data, server)
        change_to_frame(gui_frame_slider.value, frame_to_handle)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        remove_handles(frame_to_handle)
        raise


if __name__ == "__main__":
    tyro.cli(main)
