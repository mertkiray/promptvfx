"""WebGL-based Gaussian splat rendering. This is still under developmentt."""

from __future__ import annotations

import time
import tyro

from gui import Gui
from state import State
import src.viser as viser

def main() -> None:
    server = viser.ViserServer()
    state = State()
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
