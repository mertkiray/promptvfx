"""WebGL-based Gaussian splat rendering with per-user isolation."""

from __future__ import annotations
import time
import tyro
import viser

from animation import write_animation_functions
from gui import Gui
from scene import Scene
from state import State

# Dictionary to manage per-user sessions
user_sessions = {}

def handle_new_user(client: viser.ClientHandle) -> None:
    """Initialize a new session for each user when they connect."""
    print(f"New user connected: {client.client_id}")

    # Create a fresh scene, state, and GUI for this user
    user_scene = Scene(client.scene)
    user_state = State(user_scene, client.gui)
    user_gui = Gui(client, user_state, user_scene)
    user_state.attach(user_gui)

    # Store session
    user_sessions[client.client_id] = {
        "scene": user_scene,
        "state": user_state,
        "gui": user_gui
    }

def handle_disconnect(client: viser.ClientHandle) -> None:
    """Cleanup the user's session when they disconnect."""
    print(f"User {client.client_id} disconnected")

    # Remove user session if it exists
    session = user_sessions.pop(client.client_id, None)
    if session:
        session["state"].remove_gs_handles()  # Remove Gaussian splats

def main() -> None:
    """Start the Viser server and handle per-user sessions."""
    server = viser.ViserServer()

    # Attach event handlers for connect and disconnect
    server.on_client_connect(handle_new_user)
    server.on_client_disconnect(handle_disconnect)

    # Keep server alive
    server.sleep_forever()

if __name__ == "__main__":
    tyro.cli(main)
