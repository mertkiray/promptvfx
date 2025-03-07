import viser
import viser.transforms as tf
import numpy as np
import time
import tyro

from animation import write_animation_functions
from gui import Gui
from scene import Scene
from state import State

# Global dictionary to keep track of image node handles.
image_handles = {}

def place_cameras_around_origin(server: viser.ViserServer, radius=3.0, tilt_angle=30, num_cameras=8):
    """
    Place camera frustums on a circle around the origin.

    - Cameras are placed at (r*cosθ, r*sinθ, h) where h = r * tan(tilt_angle)/2
    - Each camera looks toward (0, 0, 0)
    - Returns a list of dictionaries containing each camera's parameters.
    """
    camera_params_list = []
    tilt_rad = np.radians(tilt_angle)
    h = radius * np.tan(tilt_rad) / 2
    world_up = np.array([0, 0, 1])

    for i in range(num_cameras):
        theta = 2 * np.pi * i / num_cameras
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = h
        position = np.array([x, y, z])

        look_at = np.array([0, 0, 0])
        forward = look_at - position
        forward /= np.linalg.norm(forward)

        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        adjusted_up = np.cross(forward, right)

        rotation_matrix = np.column_stack([right, adjusted_up, forward])
        wxyz = tf.SO3.from_matrix(rotation_matrix).wxyz

        server.scene.add_camera_frustum(
            name=f"camera_{i}",
            wxyz=wxyz,
            position=position,
            fov=45,
            aspect=1.5,
            scale=0.5,
            color=(1.0, 0.0, 0.0),
            visible=False
        )

        camera_params_list.append({
            'name': f"camera_{i}",
            'position': position,
            'wxyz': wxyz,
            'fov': 45,
        })

    return camera_params_list

def render_and_add_images(server: viser.ViserServer, camera_params_list, height=480, width=640, near_distance=1.0):
    clients = server.get_clients()
    if not clients:
        print("No client connected for rendering.")
        return
    client = next(iter(clients.values()))
    global image_handles

    for cam in camera_params_list:
        # Remove any existing image node with this name.
        server.scene.remove_by_name(f"image_{cam['name']}")
        if cam['name'] in image_handles:
            image_handles[cam['name']].remove()
            del image_handles[cam['name']]

        # Render a new image.
        image = client.get_render(
            height, width,
            wxyz=cam['wxyz'],
            position=cam['position'],
            fov=cam['fov']
        )
        forward = (np.array([0, 0, 0]) - cam['position'])
        forward /= np.linalg.norm(forward)
        screen_position = cam['position'] + forward * near_distance

        fov_rad = np.radians(cam['fov'])
        half_height = near_distance * np.tan(fov_rad / 2)
        render_height = 2 * half_height
        render_width = render_height * 1.5

        # Add the new image node.
        handle = server.scene.add_image(
            name=f"image_{cam['name']}",
            image=image,
            render_width=render_width / 2,  # adjust scaling as needed
            render_height=render_height / 2,
            format='jpeg',
            jpeg_quality=90,
            wxyz=cam['wxyz'],
            position=screen_position,
            visible=True
        )
        image_handles[cam['name']] = handle
        print(f"Re-rendered image for {cam['name']} at {screen_position}")

def main() -> None:
    server = viser.ViserServer()
    scene = Scene(server.scene)
    camera_params_list = place_cameras_around_origin(server, num_cameras=6)
    state = State(scene, server.gui)
    gui = Gui(server, state, scene)
    gui.camera_params_list = camera_params_list
    state.attach(gui)

    # Wait for a client to connect.
    time.sleep(5)
    # Initial render.
    render_and_add_images(server, camera_params_list)

    try:
        while True:
            time.sleep(10)
    finally:
        state.remove_gs_handles()
        write_animation_functions()

if __name__ == "__main__":
    tyro.cli(main)
