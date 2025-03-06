import numpy as np
import time
import os
import imageio
import src.viser.transforms as tf
from viser import ClientHandle, GuiApi
from state import State
from text_utils import snake_case

SLEEP = 1


class RendererOrbit:
    def __init__(self, client: ClientHandle, gui_api: GuiApi, state: State):
        self.client = client
        self.gui_api = gui_api
        self.state = state
        self.num_views = 100  # Number of discrete views for the 360° rotation

    def render_animation(self) -> None:
        status = self.gui_api.add_markdown("*Saving Frames...*")
        progress = self.gui_api.add_progress_bar(10, animated=True)
        fps = 24
        self.state.fps = fps
        self.state.visible_frame = 0

        total_frames = self.state.active_animation.duration * fps
        frames_per_next = max(1, round(self.num_views / total_frames))

        print(f"Total Frames: {total_frames}, Num Views: {self.num_views}, Frames per next_frame: {frames_per_next}")

        rendered_images: list[np.ndarray] = []

        # Get the current camera position and direction
        current_position = self.client.camera.position
        current_wxyz = self.client.camera.wxyz  # Quaternion representation of rotation

        # Convert quaternion to rotation matrix
        current_R = tf.SO3(current_wxyz).as_matrix()

        # Get the look-at point (where the camera is looking)
        lookat_vector = current_R[:, 2]  # Forward vector (z-axis)
        lookat_point = current_position + lookat_vector  # Compute look-at target

        # Compute the radius (distance from camera to look-at point)
        camera_radius = np.linalg.norm(current_position - lookat_point)

        # 2️⃣ Generate positions for a 360° orbit around the current position
        sampled_positions = self._sample_orbit(camera_radius, self.num_views, current_position, lookat_point)

        # 3️⃣ Compute camera orientations
        camera_poses = []
        for t in sampled_positions:
            lookat = lookat_point - t  # Look at the same target
            R = self._compute_rotation_matrix(lookat, np.array([0, 0, 1]))
            c2w = np.hstack((R, t.reshape(3, 1)))
            c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
            camera_poses.append(c2w)

        print(camera_poses)


        # 4️⃣ Render each discrete view
        for i, pose in enumerate(camera_poses):
            self._set_camera_pose(pose)

            if i % frames_per_next == 0:
                self.state.next_frame()

            time.sleep(SLEEP)
            rendered_images.append(self.client.get_render(height=1080, width=1920))
            progress.value += 100 / total_frames

        # 5️⃣ Save images
        status.content = "*Writing JPGs...*"
        progress.value += 10
        animation = self.state.active_animation
        subdirectory = render_subdirectory(animation.title, animation.duration)
        output_dir = f"render/{subdirectory}/final/{fps}"
        self._write_images(images=rendered_images, output_dir=output_dir)
        progress.remove()
        status.remove()

    def _sample_orbit(self, radius, num_views, current_position, lookat_point):
        """Generate camera positions for a 360° orbit around the current position."""
        phi_values = np.linspace(0, 2 * np.pi, num_views)

        # Use the same elevation angle
        theta_rad = np.arcsin((current_position[2] - lookat_point[2]) / radius)

        x = np.cos(theta_rad) * np.cos(phi_values) * radius + lookat_point[0]
        y = np.cos(theta_rad) * np.sin(phi_values) * radius + lookat_point[1]
        z = np.sin(theta_rad) * radius + lookat_point[2]

        return np.stack((x, y, z), axis=-1)

    def _compute_rotation_matrix(self, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute camera rotation matrix from OpenCV to OpenGL."""
        z_axis = lookat / np.linalg.norm(lookat)
        x_axis = np.cross(z_axis, up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        return np.array((x_axis, y_axis, z_axis)).T

    def _set_camera_pose(self, pose: np.ndarray) -> None:
        """Set camera to a new pose instantly."""
        with self.client.atomic():
            self.client.camera.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
            self.client.camera.position = pose[:3, 3]

    def _write_images(self, images, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(images):
            imageio.imwrite(f"{output_dir}/img_{idx}.jpg", image)

def render_subdirectory(animation_title: str, animation_duration: int) -> str:
    return f"{snake_case(animation_title)}_{animation_duration}s"
