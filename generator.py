import time
from dataclasses import dataclass

from animation import Animation, AnimationEvolution, VisionAngle
from llm_utils import (
    generate_abstract_summary,
    generate_animation_score,
    generate_auto_improved_centers_code,
    generate_auto_improved_opacities_code,
    generate_auto_improved_rgbs_code,
    generate_centers_behavior,
    generate_centers_code,
    generate_feedback_improved_centers_code,
    generate_feedback_improved_opacities_code,
    generate_feedback_improved_rgbs_code,
    generate_opacities_behavior,
    generate_opacities_code,
    generate_rgbs_behavior,
    generate_rgbs_code,
)
from renderer import Renderer, render_subdirectory
from state import State
from viser import ClientHandle, GuiApi


@dataclass
class GeneratorConfig:
    animation_title: str
    animation_description: str
    animation_duration: int
    design_temperature: float
    code_temperature: float
    vision_angles: list[VisionAngle]
    n_samples: int
    n_improves: int


class Generator:
    def __init__(
        self,
        config: GeneratorConfig,
        client: ClientHandle,
        gui_api: GuiApi,
        state: State,
    ):
        self.config = config
        self.client = client
        self.gui_api = gui_api
        self.state = state
        self.output = AnimationEvolution()
        self.subdirectory = render_subdirectory(
            config.animation_title, config.animation_duration
        )

    def run(self) -> None:
        self._auto_sample()
        self._auto_improve()

    def apply_feedback(self, feedback: str) -> None:
        animation = self.output.final_animation
        code_temperature = self.config.code_temperature
        description = animation.description
        centers_code = animation.centers_code
        rgbs_code = animation.rgbs_code
        opacities_code = animation.opacities_code

        status = self.gui_api.add_markdown("")

        status.content = "*Rendering Current Snapshots*"
        progress = self.gui_api.add_progress_bar(value=100, animated=True)
        self.state.active_animation = animation
        time.sleep(1)
        self.state.visible_frame = 0
        time.sleep(1)
        renderer = Renderer(self.client, self.gui_api, self.state)
        image_dir = f"render/{self.subdirectory}/feedback/input_{len(self.output.feedback_to_animation)}"
        first_frames = renderer.render_first_frames(8, image_dir)

        ## Step 2: Improve `compute_centers`
        status.content = "*Improving `compute_centers`*"
        progress.value = 1
        centers_code = animation.centers_code
        improved_centers_code = ""
        while True:
            improved_centers_code = generate_feedback_improved_centers_code(
                feedback, centers_code, first_frames, description, code_temperature
            )
            if self._test_centers_code(improved_centers_code):
                break
        ## Step 3: Improve `compute_rgbs`
        status.content = "*Improving `compute_rgbs`...*"
        progress.value = 34
        rgbs_code = animation.rgbs_code
        improved_rgbs_code = ""
        while True:
            improved_rgbs_code = generate_feedback_improved_rgbs_code(
                feedback, rgbs_code, first_frames, description, code_temperature
            )
            if self._test_rgbs_code(improved_rgbs_code):
                break
        ## Step 4: Improve `compute_opacities`
        status.content = "*Improving `compute_opacities`...*"
        progress.value = 67
        opacities_code = animation.opacities_code
        improved_opacities_code = ""
        while True:
            improved_opacities_code = generate_feedback_improved_opacities_code(
                feedback, opacities_code, first_frames, description, code_temperature
            )
            if self._test_opacities_code(improved_opacities_code):
                break
        progress.value = 100

        progress.remove()
        status.remove()

        animation_with_feedback = Animation(
            title=animation.title,
            abstract_summary=animation.abstract_summary,
            centers_behavior=animation.centers_behavior,
            rgbs_behavior=animation.rgbs_behavior,
            opacities_behavior=animation.opacities_behavior,
            centers_code=improved_centers_code,
            rgbs_code=improved_rgbs_code,
            opacities_code=improved_opacities_code,
        )

        self.output.feedback_to_animation[feedback] = animation_with_feedback
        self.output.final_animation = animation_with_feedback

    def _auto_sample(self) -> None:
        status = self.gui_api.add_markdown("")
        vision_images = self._render_vision_angles()
        n_samples = self.config.n_samples
        for i in range(n_samples):
            status.content = f"*Generating Sample {i + 1}/{n_samples}*"
            sample = self._generate_animation(vision_images)
            self.output.auto_sampled_animations.append(sample)
            if n_samples > 1:
                image_dir = f"render/{self.subdirectory}/auto_sample/sample_{i}"
                score = self._score_animation(sample, image_dir)
                sample.score = score
        status.remove()

    def _auto_improve(self) -> None:
        best_sample = self._get_best_sample()
        n_improves = self.config.n_improves
        if n_improves <= 0:
            self.output.final_animation = best_sample

        status = self.gui_api.add_markdown("")
        sub_status = self.gui_api.add_markdown("")
        progress = self.gui_api.add_progress_bar(100, animated=True)

        description = self.config.animation_description
        code_temperature = self.config.code_temperature

        improved_animation = best_sample
        for i in range(n_improves):
            status.content = f"*Auto Improve Iteration {i + 1}/{n_improves}*"
            ## Step 1: Render Current Animation
            sub_status.content = "*Rendering Current Snapshots*"
            self.state.active_animation = improved_animation
            time.sleep(1)
            self.state.visible_frame = 0
            time.sleep(1)
            renderer = Renderer(self.client, self.gui_api, self.state)
            image_dir = f"render/{self.subdirectory}/auto_improve/input_{i}"
            first_frames = renderer.render_first_frames(8, image_dir)
            ## Step 2: Improve `compute_centers`
            sub_status.content = "*Improving `compute_centers`*"
            centers_code = improved_animation.centers_code
            improved_centers_code = ""
            while True:
                improved_centers_code = generate_auto_improved_centers_code(
                    centers_code, first_frames, description, code_temperature
                )
                if self._test_centers_code(improved_centers_code):
                    break
            ## Step 3: Improve `compute_rgbs`
            sub_status.content = "*Improving `compute_rgbs`...*"
            rgbs_code = improved_animation.rgbs_code
            improved_rgbs_code = ""
            while True:
                improved_rgbs_code = generate_auto_improved_rgbs_code(
                    rgbs_code, first_frames, description, code_temperature
                )
                if self._test_rgbs_code(improved_rgbs_code):
                    break
            ## Step 4: Improve `compute_opacities`
            sub_status.content = "*Improving `compute_opacities`...*"
            opacities_code = improved_animation.opacities_code
            improved_opacities_code = ""
            while True:
                improved_opacities_code = generate_auto_improved_opacities_code(
                    opacities_code, first_frames, description, code_temperature
                )
                if self._test_opacities_code(improved_opacities_code):
                    break
            # Set Impoved Animation
            improved_animation = Animation(
                title=best_sample.title,
                description=best_sample.description,
                duration=best_sample.duration,
                abstract_summary=best_sample.abstract_summary,
                centers_behavior=best_sample.centers_behavior,
                rgbs_behavior=best_sample.rgbs_behavior,
                opacities_behavior=best_sample.opacities_behavior,
                centers_code=improved_centers_code,
                rgbs_code=improved_rgbs_code,
                opacities_code=improved_opacities_code,
            )
            self.output.auto_improved_animations.append(improved_animation)

        self.output.final_animation = improved_animation

        progress.remove()
        sub_status.remove()
        status.remove()

    def _generate_animation(self, images: list[str] = []) -> Animation:
        title = self.config.animation_title
        description = self.config.animation_description
        duration = self.config.animation_duration
        design_temperature = self.config.design_temperature
        code_temperature = self.config.code_temperature

        # Design Phase
        status = self.gui_api.add_markdown("(1/7) *Interpreting Description...*")
        progress = self.gui_api.add_progress_bar(value=12, animated=True)
        abstract_summary = generate_abstract_summary(
            description, duration, design_temperature, images
        )

        status.content = "(2/7) *Designing Centers Behavior...*"
        progress.value = 24
        centers_behavior = generate_centers_behavior(
            abstract_summary, design_temperature, images
        )

        status.content = "(3/7) *Designing RGBs Behavior...*"
        progress.value = 36
        rgbs_behavior = generate_rgbs_behavior(
            abstract_summary, design_temperature, images
        )

        status.content = "(4/7) *Designing Opacities Behavior...*"
        progress.value = 48
        opacities_behavior = generate_opacities_behavior(
            abstract_summary, design_temperature, images
        )

        # Code Phase
        status.content = "(5/7) *Generating `compute_centers`...*"
        progress.value = 60
        centers_code = ""
        while True:
            centers_code = generate_centers_code(
                centers_behavior, duration, code_temperature, images
            )
            if self._test_centers_code(centers_code):
                break

        status.content = "(6/7) *Generating `compute_rgbs`...*"
        progress.value = 72
        rgbs_code = ""
        while True:
            rgbs_code = generate_rgbs_code(
                rgbs_behavior, duration, code_temperature, images
            )
            if self._test_rgbs_code(rgbs_code):
                break

        status.content = "(7/7) *Generating `compute_opacities`...*"
        progress.value = 85
        opacities_code = ""
        while True:
            opacities_code = generate_opacities_code(
                opacities_behavior, duration, code_temperature, images
            )
            if self._test_opacities_code(opacities_code):
                break

        progress.remove()
        status.remove()

        return Animation(
            title,
            description,
            duration,
            abstract_summary,
            centers_behavior,
            rgbs_behavior,
            opacities_behavior,
            centers_code,
            rgbs_code,
            opacities_code,
        )

    def _render_vision_angles(self) -> list[str]:
        angles = self.config.vision_angles
        renderer = Renderer(self.client, self.gui_api, self.state)
        angle_images_dir = f"render/{self.subdirectory}/vision_angles"
        angle_images = renderer.render_angles(angles, angle_images_dir)
        return angle_images

    def _score_samples(self) -> None:
        status = self.gui_api.add_markdown("")
        progress = self.gui_api.add_progress_bar(100, animated=True)

        samples = self.output.auto_sampled_animations
        for idx, animation in enumerate(samples):
            status.content = f"*Scoring Sample {idx + 1}/{len(samples)}*"
        progress.remove()
        status.remove()

    def _score_animation(self, animation: Animation, image_dir: str) -> int:
        status = self.gui_api.add_markdown("Scoring Animation")
        progress = self.gui_api.add_progress_bar(100, animated=True)
        self.state.active_animation = animation
        time.sleep(1)
        self.state.visible_frame = 0
        time.sleep(1)
        renderer = Renderer(self.client, self.gui_api, self.state)
        first_frames = renderer.render_first_frames(8, image_dir)
        score = generate_animation_score(first_frames, animation.description)
        status.remove()
        progress.remove()
        return score

    def _get_best_sample(self) -> Animation:
        score_to_animation = {
            sample.score or 0: sample for sample in self.output.auto_sampled_animations
        }
        max_score = max(score_to_animation.keys())
        return score_to_animation[max_score]

    def _test_centers_code(self, centers_code: str) -> bool:
        try:
            self.state.active_animation = Animation(centers_code=centers_code)
        except Exception:
            return False
        return True

    def _test_rgbs_code(self, rgbs_code: str) -> bool:
        try:
            self.state.active_animation = Animation(rgbs_code=rgbs_code)
        except Exception:
            return False
        return True

    def _test_opacities_code(self, opacities_code: str) -> bool:
        try:
            self.state.active_animation = Animation(opacities_code=opacities_code)
        except Exception:
            return False
        return True
