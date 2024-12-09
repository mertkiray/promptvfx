"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf
from viser._messages import GaussianSplatsMessage, GaussianSplatsProps
from viser._scene_handles import GaussianSplatHandle, colors_to_uint8


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

def update_centers(centers: np.ndarray, t: float) -> np.ndarray:
        """Input prompt:
        Given an ndarray "points" of size [N, 3] centered at the origin (0,0,0), I want 
        you to write me a python function with the signature "def update_points(points: 
        np.ndarray, t: float) -> np.ndarray:". The function should translate the given 
        points based on a time parameter t. Over time this should create an animation 
        that looks like the point cloud explodes and falls down due to gravity. Make 
        sure the animation looks as realistic and cool as possible and not like a 
        computer simulation.
        """
        gravity = np.array([0, 0, -9.81])
        damping = 0.98
        np.random.seed(0)
        initial_velocities = np.random.uniform(-5, 5, size=centers.shape)
        velocities = initial_velocities * damping ** t + gravity * t
        return centers + velocities * t

def update_rgbs(rgbs: np.ndarray, t: float) -> np.ndarray:
    """Input prompt:
    You are given an ndarray "rgbs" of shape [N, 3]. The values are floats in between
    0.0 and 1.0. The ndarray belongs to a gaussian splatting model. I want you to write
    me a python function with the signature
    "def update_rgbs(rgbs:  np.ndarray, t: float) -> np.ndarray:".
    The function should change the given rgb values based on a time parameter t which
    can also be between 0.0 and 1.0. Over time this should create an animation.
    The centers and opacities are already updated correctly over time. The animation can
    be described as follows: "The object explodes and falls down due to gravity." Make
    sure the animation looks as realistic and cool as possible and not like a  computer
    simulation.
    """
    # Ensure t is clamped between 0.0 and 1.0
    t = np.clip(t, 0.0, 1.0)

    # Simulate the "explosion" phase (0.0 <= t <= 0.5)
    if t <= 0.5:
        # Scale the brightness to peak at t = 0.5
        brightness = np.interp(t, [0.0, 0.5], [1.0, 2.0])
        colors = rgbs * brightness
        # Introduce some fiery hues (shift towards red and yellow)
        colors[:, 0] = np.clip(colors[:, 0] * 1.2, 0.0, 1.0)  # Enhance red
        colors[:, 1] = np.clip(colors[:, 1] * 0.8, 0.0, 1.0)  # Slightly dampen green
        colors[:, 2] = np.clip(colors[:, 2] * 0.5, 0.0, 1.0)  # Strongly dampen blue
    
    # Simulate the "falling and cooling" phase (0.5 < t <= 1.0)
    else:
        # Dim the colors gradually as t progresses
        brightness = np.interp(t, [0.5, 1.0], [2.0, 0.5])
        colors = rgbs * brightness
        # Shift hues towards blue and reduce intensity
        colors[:, 0] = np.clip(colors[:, 0] * 0.7, 0.0, 1.0)  # Reduce red
        colors[:, 1] = np.clip(colors[:, 1] * 0.9, 0.0, 1.0)  # Slightly reduce green
        colors[:, 2] = np.clip(colors[:, 2] * 1.3, 0.0, 1.0)  # Enhance blue

    # Normalize the colors back to the [0.0, 1.0] range
    colors = np.clip(colors, 0.0, 1.0)

    return colors

def update_opacities(opacities: np.ndarray, t: float) -> np.ndarray:
    """Input prompt:
    You are given an ndarray "opacities" of shape [N, 1]. The values are floats in
    between 0.0 and 1.0. The ndarray belongs do a gaussian splatting model. I want you
    to write me a python function with the signature
    "def update_opacities(opacities:  np.ndarray, t: float) -> np.ndarray:".
    The function should change the given opacity values based on a time parameter t
    which can also be between 0.0 and 1.0. Over time this should create an animation.
    The centers and rgbs are already updated correctly over time. The animation can be
    described as follows: "The object explodes and falls down due to gravity." Make sure
    the animation looks as realistic and cool as possible and not like a  computer
    simulation.
    """
    N = opacities.shape[0]

    # Random variations to make the effect more natural
    random_variation = np.random.uniform(0.8, 1.2, size=(N, 1))

    # Explosion phase: opacity spikes up
    explosion_opacity = np.clip((1 - t) * np.random.uniform(0.8, 1.0, size=(N, 1)), 0.0, 1.0)

    # Decay phase: opacity fades with time
    decay_factor = np.exp(-5 * t)  # Exponential decay for realism
    decay_opacity = decay_factor * random_variation

    # Combine the phases: fade from explosion to decay
    updated_opacities = explosion_opacity * (1 - t) + decay_opacity * t

    # Ensure opacities are within the valid range [0.0, 1.0]
    updated_opacities = np.clip(updated_opacities, 0.0, 1.0)

    return updated_opacities

def add_splat_at_t(t: float, server: viser.ViserServer, splat_data: SplatFile, ):
    if np.isclose(t, 0.0):
        return server.scene.add_gaussian_splats(
        "gaussian_splat",
        centers=splat_data["centers"],
        rgbs=splat_data["rgbs"],
        opacities=splat_data["opacities"],
        covariances=splat_data["covariances"],
        )
    
    updated_centers = update_centers(splat_data["centers"], t)
    updated_rgbs = update_rgbs(splat_data["rgbs"], t)
    updated_opacities = update_opacities(splat_data["opacities"], t)

    return server.scene.add_gaussian_splats(
        "gaussian_splat",
        centers=updated_centers,
        rgbs=updated_rgbs,
        opacities=updated_opacities,
        covariances=splat_data["covariances"],
    )

def main(splat_path: Path) -> None:
    server = viser.ViserServer()

    if splat_path.suffix == ".splat":
        splat_data = load_splat_file(splat_path, center=True)
    elif splat_path.suffix == ".ply":
        splat_data = load_ply_file(splat_path, center=True)
    else:
        raise SystemExit("Please provide a filepath to a .splat or .ply file.")

    server.scene.add_transform_controls("0",scale=0.3, opacity=0.3)
    gs_handle = server.scene.add_gaussian_splats(
        "gaussian_splat",
        centers=splat_data["centers"],
        rgbs=splat_data["rgbs"],
        opacities=splat_data["opacities"],
        covariances=splat_data["covariances"],
    )

    server.gui.configure_theme(dark_mode=True)

    gui_name = server.gui.add_markdown("Current Animation: **None**")

    gui_generator = server.gui.add_button(label="Prompt", icon=viser.Icon.ROBOT)
    @gui_generator.on_click
    def _(_) -> None:
        with server.gui.add_modal(title="Animation Generator 🤖") as gui_popout:
            gui_popout_name = server.gui.add_text("Name", "New Animation")
            gui_popout_input = server.gui.add_text("Prompt", "")

            gui_popout_generate = server.gui.add_button("Generate 🛠")
            @gui_popout_generate.on_click
            def _(_) -> None:
                if not gui_popout_input.value:
                    return
                # TODO Generate functions with LLM
                gui_name.content = f"Current Animation: **{gui_popout_name.value}**"
                gui_popout.close()

            gui_popout_abort = server.gui.add_button("", icon=viser.Icon.X, color="red")
            @gui_popout_abort.on_click
            def _(_) -> None:
                gui_popout.close()
    
    

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_slider = server.gui.add_slider(
        label="t",
        min=0.0,
        max=1.0,
        step=0.1,
        initial_value=0.0,
    )


    @gui_slider.on_update
    def _(_) -> None:
        nonlocal gs_handle
        t = gui_slider.value
        gs_handle.remove()
        time.sleep(0.1)
        gs_handle = add_splat_at_t(t, server, splat_data)

    gui_play = server.gui.add_button(
        label="Play",
        color="green",
        icon=viser.Icon.BRAND_GOOGLE_PLAY
    )

    @gui_play.on_click
    def _(_) -> None:
        nonlocal gs_handle
        gui_play.disabled = True
        animation_stopped = False

        gui_stop = server.gui.add_button("Stop", color="red")
        @gui_stop.on_click
        def _(_) -> None:
            nonlocal animation_stopped
            animation_stopped=True
            gui_stop.remove()
            gui_play.disabled = False

        for i in range(int(gui_slider.value * 10), int(gui_slider.max * 10)+1):
            if animation_stopped:
                return
            t = i / 10.0
            gui_slider.value = t
            gs_handle.remove()
            time.sleep(0.1)
            gs_handle = add_splat_at_t(t, server, splat_data)
            time.sleep(1)
        
        gui_slider.value = 0
        gui_stop.remove()
        gui_play.disabled = False

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
