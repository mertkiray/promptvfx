"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
import tyro

from gui import Gui
from state import State
import viser
from splat_utils import load_splat

def main(splat_path: Path) -> None:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    state = State(splat_path)
    gui = Gui(server, state)
    state.attach(gui)
    bg_data = load_splat(Path("data/garden_bg.splat"))
    bg_handle = server.scene.add_gaussian_splats(
            f"splat_bg",
            centers=bg_data["centers"],
            rgbs=bg_data["rgbs"],
            opacities=bg_data["opacities"],
            covariances=bg_data["covariances"],
            )
    bg_handle.position = (0.2, 0.2, -1.75)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        state.remove_gs_handles()
        raise


if __name__ == "__main__":
    tyro.cli(main)
