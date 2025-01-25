"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
from pathlib import Path
import tyro

from gui import Gui
from state import State
import viser


def main(splat_path: Path) -> None:
    server = viser.ViserServer()
    server.scene.add_transform_controls("0",scale=0.3, opacity=0.3)

    state = State(splat_path, fps=2)
    gui = Gui(server, state)
    state.attach(gui)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        state.remove_gs_handles()
        raise


if __name__ == "__main__":
    tyro.cli(main)
