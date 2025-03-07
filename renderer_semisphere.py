import numpy as np
import time
import os
import imageio
import src.viser.transforms as tf
from viser import ClientHandle, GuiApi
from state import State
from text_utils import snake_case

SLEEP = 1

# Camera position sampling function
def grid_half_sphere(radius=1.5, num_views=30, theta=None, phi_range=(0, 360)):
    if theta is None:
        theta = np.deg2rad(np.array((0, 15, 30, 45, 60)))
    else:
        theta = np.deg2rad([theta])
    phi = np.deg2rad(np.linspace(phi_range[0], phi_range[1], num_views // len(theta) + 1)[:-1])
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()
    x = np.cos(theta) * np.cos(phi) * radius
    y = np.cos(theta) * np.sin(phi) * radius
    z = np.sin(theta) * radius
    t = np.stack((x, y, z), axis=-1)
    return t


class RendererSemisphere:
    def __init__(self, client: ClientHandle, gui_api: GuiApi, state: State, server, camera_params_list):
        self.client = client
        self.gui_api = gui_api
        self.state = state
        self.camera_radius = 3  # Camera radius
        self.num_views = 360  # Number of discrete views
        self.theta_angle = 30  # Max elevation angle in degrees
        self.phi_range = (0, 360)  # Full 360-degree rotation
        self.server=server
        self.camera_params_list = camera_params_list

    def render_animation(self) -> None:
        status = self.gui_api.add_markdown("*Saving Frames...*")
        progress = self.gui_api.add_progress_bar(10, animated=True)
        fps = 24
        self.state.fps = fps
        self.state.visible_frame = 0

        # Compute total frames for animation
        total_frames = self.state.active_animation.duration * fps

        # Compute how often to call `next_frame`
        frames_per_next = max(1, round(self.num_views / total_frames))  # Determines frame update rate

        print(f"Total Frames: {total_frames}, Num Views: {self.num_views}, Frames per next_frame: {frames_per_next}")

        rendered_images: list[np.ndarray] = []

        # Sample discrete camera positions
        center = np.array([0, 0, 0])
        sampled_positions = grid_half_sphere(self.camera_radius, self.num_views, self.theta_angle,
                                             self.phi_range) + center

        # Compute camera orientations
        camera_poses = []
        for t in sampled_positions:
            lookat = center - t
            R = self._compute_rotation_matrix(lookat, np.array([0, 0, 1]))
            c2w = np.hstack((R, t.reshape(3, 1)))
            c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
            camera_poses.append(c2w)

        # Render each discrete view, adjusting for animation frames
        for i, pose in enumerate(camera_poses):
            self._set_camera_pose(pose)
            time.sleep(SLEEP*1)  # Allow time for rendering

            # Call `next_frame()` every `frames_per_next` views
            if i % frames_per_next == 0:
                self.state.next_frame()

            time.sleep(SLEEP*1)  # Allow time for rendering
            rendered_images.append(self.client.get_render(height=1080, width=1920))
            progress.value += 100 / total_frames  # Update progress bar
            time.sleep(SLEEP*2)

            from main import render_and_add_images  # adjust import if necessary
            render_and_add_images(self.server, self.camera_params_list)
            time.sleep(SLEEP*2)

        # Write images to disk
        status.content = "*Writing JPGs...*"
        progress.value += 10
        animation = self.state.active_animation
        subdirectory = render_subdirectory(animation.title, animation.duration)
        output_dir = f"render/{subdirectory}/final/{fps}"
        self._write_images(images=rendered_images, output_dir=output_dir)
        progress.remove()
        status.remove()

    def _compute_rotation_matrix(self, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute camera rotation matrix from a look-at direction."""
        z_axis = lookat / np.linalg.norm(lookat)
        x_axis = np.cross(z_axis, up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.array((x_axis, y_axis, z_axis)).T
        return R

    def _set_camera_pose(self, pose: np.ndarray) -> None:
        """Set camera to a new pose instantly without interpolation."""
        with self.client.atomic():
            self.client.camera.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
            self.client.camera.position = pose[:3, 3]

    def _write_images(self, images, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(images):
            imageio.imwrite(f"{output_dir}/img_{idx}.jpg", image)

def render_subdirectory(animation_title: str, animation_duration: int) -> str:
    return f"{snake_case(animation_title)}_{animation_duration}s"
