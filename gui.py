from pathlib import Path
import tempfile
import re
import time
import traceback
import numpy as np
from splat_utils import load_splat
from state import State, Observer
import src.viser as viser
from src.viser._gui_api import GuiApi
from src.viser._scene_api import SceneApi
from src.viser._viser import ViserServer
from llm_utils import AnimationSummary, prompt_4o,  prompt_o1, extract_centers_summary, extract_rgbs_summary, extract_opacities_summary
import prompts


# Default Functions
def compute_centers(t: float, centers: np.ndarray) -> np.ndarray:
    return centers

def compute_rgbs(t: float, rgbs: np.ndarray) -> np.ndarray:
    return rgbs

def compute_opacities(t: float, opacities: np.ndarray) -> np.ndarray:
    return opacities

class Gui(Observer):
    def __init__(self, server: ViserServer, state: State):
        self.scene_api: SceneApi = server.scene
        self.gui_api: GuiApi = server.gui
        self.gui_api.configure_theme(dark_mode=True)
        self.state: State = state
        self.load_background()
        self.state.frame_to_gs_handle[0] = self.add_splat_at_t(0)

        # Gui Elements
        self.tab_group = self.gui_api.add_tab_group()
        ## Animation Tab
        with self.tab_group.add_tab("Animation"):
            self.title_md = self.gui_api.add_markdown(f"Title: **{self.state.animation_title}**")

            self.prompt_btn = self.gui_api.add_button("Prompt", icon=viser.Icon.ROBOT)
            self.prompt_btn.on_click(lambda _: self.open_prompt())

            self.inspect_btn = self.gui_api.add_button("Functions", icon=viser.Icon.FUNCTION)
            self.inspect_btn.on_click(lambda _: self.open_inspector())

            self.fps_btn_grp = self.gui_api.add_button_group(f"FPS ({self.state.fps})", ["8", "24", "60"])
            self.fps_btn_grp.on_click(lambda _: self.update_fps())

            self.speed_btn_grp = self.gui_api.add_button_group(
                label=f"Speed ({self.state.speed})",
                options=["0.25x", "0.5x", "1x", "2x"])
            self.speed_btn_grp.on_click(lambda _: self.update_speed())

            self.frame_slider = self.gui_api.add_slider(
                label="Current Frame",
                min=0,
                max=self.state.total_frames-1,
                step=1,
                initial_value=0,
            )
            self.frame_slider.on_update(lambda _: self.update_frame())
                

            self.play_btn = server.gui.add_button("▶ Play", color="green")
            self.play_btn.on_click(lambda _: self.play())
        ## Scene Tab
        with self.tab_group.add_tab("Scene"):
            self.world_axis_cbox = self.gui_api.add_checkbox("World Axis", initial_value=False)
            self.world_axis_cbox.on_update(lambda _: self.toggle_world_axis())

            self.background_cbox = self.gui_api.add_checkbox("Background", initial_value=True)
            self.background_cbox.on_update(lambda _: self.toggle_background())
            
            with self.gui_api.add_folder("Splat Files"):
                self.luigi_btn = self.gui_api.add_button("luigi.splat")
                self.luigi_btn.on_click(lambda _: self.change_to_splat(Path("data/luigi.splat")))

                self.nike_btn = self.gui_api.add_button("nike.splat")
                self.nike_btn.on_click(lambda _: self.change_to_splat(Path("data/nike.splat")))

                self.vase_btn = self.gui_api.add_button("vase.splat")
                self.vase_btn.on_click(lambda _: self.change_to_splat(Path("data/vase.splat")))

                self.upload_btn = self.gui_api.add_upload_button("Upload", icon=viser.Icon.UPLOAD, color="teal")
                self.upload_btn.on_upload(lambda _: self.process_upload())

    def update(self):
        self.title_md.content = f"Title: **{self.state.animation_title} ({self.state.animation_duration}s)**"
        self.frame_slider.max = self.state.total_frames-1
        self.fps_btn_grp.label = f"FPS ({self.state.fps})"
        self.speed_btn_grp.label = f"Speed ({self.state.speed})"
        self.fps_btn_grp.disabled = False

    def process_upload(self):
        file = self.upload_btn.value
        with tempfile.NamedTemporaryFile(suffix=file.name) as temp_file:
            temp_file.write(file.content)
            temp_file_path = Path(temp_file.name)
            self.change_to_splat(temp_file_path)

    def terminate_animation(self):
        self.state.animation_running = False
        self.frame_slider.value = 0
        self.update_frame()

    def change_to_splat(self, splat_path: Path):
        self.terminate_animation()
        self.state.splat_path = splat_path
        self.reload_splats()

    def toggle_background(self):
        self.state.background.visible = not self.state.background.visible

    def load_background(self):
        bg_data = load_splat(Path("data/table.splat"))
        bg_handle = self.scene_api.add_gaussian_splats(
                f"splat_bg",
                centers=bg_data["centers"],
                rgbs=bg_data["rgbs"],
                opacities=bg_data["opacities"],
                covariances=bg_data["covariances"],
                )
        bg_handle.position = (0.2, 0.2, -1.75)
        self.state.background = bg_handle

    def toggle_world_axis(self):
        self.scene_api.world_axes.visible = not self.scene_api.world_axes.visible

    def update_speed(self):
        self.state.speed = self.speed_btn_grp.value

    def update_fps(self):
        self.terminate_animation()
        self.state.fps = int(self.fps_btn_grp.value)
        self.reload_splats()

    def play(self):
        self.play_btn.disabled = True
        # self.fps_btn_grp.disabled = True
        self.play_animation()
        self.play_btn.disabled = False
        # self.fps_btn_grp.disabled = False


    def update_frame(self):
        self.state.change_to_frame(self.frame_slider.value)

    def open_debug(self, error_msg: str, t: float) -> None:
        with self.gui_api.add_modal("🐞 Debugger") as popout:
            self.gui_api.add_markdown(
                f"At t=**{np.round(t, 3)}** the following error occurred:\n```{error_msg}\n```")
            
            close_btn = self.gui_api.add_button("⨯", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()


    def add_splat_at_t(self, t: float):
        if np.isclose(t, 0.0):
            return self.scene_api.add_gaussian_splats(
            f"splat_{self.state.splat_path.stem}_0",
            centers=self.state._splat["centers"],
            rgbs=self.state._splat["rgbs"],
            opacities=self.state._splat["opacities"],
            covariances=self.state._splat["covariances"],
            )
        
        try:
            centers = self.state._splat["centers"]
            rgbs = self.state._splat["rgbs"]
            opacities = self.state._splat["opacities"]
            centers_at_t = globals()["compute_centers"](t, centers)
            rbgs_at_t = globals()["compute_rgbs"](t, rgbs)
            opacities_at_t = globals()["compute_opacities"](t, opacities)
            return self.scene_api.add_gaussian_splats(
                f"splat_{self.state.splat_path.stem}_{t}",
                centers=centers_at_t,
                rgbs=rbgs_at_t,
                opacities=opacities_at_t,
                covariances=self.state._splat["covariances"],
            )
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.open_debug(error_msg=stack_trace, t=t)
            raise

    def load_splats(self) -> None:
        loading_md = self.gui_api.add_markdown("*Loading Frames...*")
        progress_bar = self.gui_api.add_progress_bar(0.0, animated=True)
        seconds_per_frame = 1.0 / self.state.fps
        for frame in range(self.state.total_frames):
            t = frame * seconds_per_frame
            try:
                gs_handle = self.add_splat_at_t(t)
                gs_handle.visible = False
                self.state.frame_to_gs_handle[frame] = gs_handle
                progress_bar.value = ((frame+1)/self.state.fps) * 100
            except Exception:
                break
            
            if not self.state.has_animation:
                break
        self.state.change_to_frame(self.frame_slider.value)
        progress_bar.remove()
        loading_md.remove()

    def reload_splats(self) -> None:
        self.state.remove_gs_handles()
        self.load_splats()
        self.state.change_to_frame(self.frame_slider.value)

    def as_code_block(self, text: str) -> str:
        return f"```\n{text}\n```"
    
    

    def extract_python_code(self, markdown: str):
        pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        match = pattern.search(markdown)
        return match.group(1) if match else ""

    def open_generator(self):
        with self.gui_api.add_modal(title="🤖 Animation Generator") as popout:
            description_txt = self.gui_api.add_text("Description", self.state.animation_description)
            temperatur_slider = self.gui_api.add_slider(
                label="Temperature",
                min=0.00,
                max=2.00,
                step=0.01,
                initial_value=self.state.temperature
            )
            duration_slider = self.gui_api.add_slider(
                label="Duration (s)",
                min=1,
                max=10,
                step=1,
                initial_value=self.state.animation_duration
            )

            generate_btn = self.gui_api.add_button("🛠 Generate")
            @generate_btn.on_click
            def _(_) -> None:
                if not description_txt.value:
                    return
                popout.close()

                self.state.animation_description = description_txt.value
                self.state.temperature = temperatur_slider.value
                self.state.animation_duration = duration_slider.value

                centers_fn, rgbs_fn, opacities_fn = self.generate_functions()
                exec(centers_fn, globals())
                exec(rgbs_fn, globals())
                exec(opacities_fn, globals())

                self.state.centers_fn_md = self.as_code_block(centers_fn)
                self.state.rgbs_fn_md = self.as_code_block(rgbs_fn)
                self.state.opacities_fn_md = self.as_code_block(opacities_fn)
                self.state.has_animation = True

                self.reload_splats()

            close_btn = self.gui_api.add_button("⨯", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()

    def open_feedback(self):
        with self.gui_api.add_modal(title="🗨 Feedback Loop") as popout:
            input_txt = self.gui_api.add_text("Feedback", "")

            improve_btn = self.gui_api.add_button("⮞ Submit")
            @improve_btn.on_click
            def _(_) -> None:
                if not input_txt.value:
                    return
                popout.close()
                centers_fn_code, rgbs_fn_code, opacities_fn_code = self.generate_functions_with_feedback(input_txt.value)
                self.state.centers_fn_md = self.as_code_block(centers_fn_code)
                self.state.rgbs_fn_md = self.as_code_block(rgbs_fn_code)
                self.state.opacities_fn_md = self.as_code_block(opacities_fn_code)
                self.reload_splats()

            
            generator_btn = self.gui_api.add_button(label="♻ New Animation",  color="green")
            @generator_btn.on_click
            def _(_) -> None:
                popout.close()
                self.open_generator()

            close_btn = self.gui_api.add_button("⨯", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()

    def open_prompt(self):
        if self.state.has_animation:
                self.open_feedback()
        else:
                self.open_generator()

    def open_inspector(self):
        with self.gui_api.add_modal("🔎 Function Inspector") as popout:
            self.gui_api.add_markdown("Centers")
            self.gui_api.add_markdown(self.state.centers_fn_md)
            self.gui_api.add_markdown("---")
            self.gui_api.add_markdown("RGBs")
            self.gui_api.add_markdown(self.state.rgbs_fn_md)
            self.gui_api.add_markdown("---")
            self.gui_api.add_markdown("Opacities")
            self.gui_api.add_markdown(self.state.opacities_fn_md)

            close_btn = self.gui_api.add_button("⨯", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()

    def play_animation(self):
        self.state.animation_running = True

        stop_btn = self.gui_api.add_button("◼ Stop", color="red")
        @stop_btn.on_click
        def _(_) -> None:
            self.state.animation_running = False

        while True:
            for frame in range(self.frame_slider.value+1, self.state.total_frames):
                if not self.state.animation_running:
                    stop_btn.remove()
                    return
                self.state.change_to_frame(frame)
                self.frame_slider.value = frame
                time.sleep((1.0/self.state.fps) * self.speed_to_sleep_factor(self.state.speed))
            self.state.change_to_frame(0)
            self.frame_slider.value = 0
            time.sleep((1.0/self.state.fps) * self.speed_to_sleep_factor(self.state.speed))

    def speed_to_sleep_factor(self, speed: str) -> float:
        if speed == "0.25x":
            return 4.0
        elif speed == "0.5x":
            return 2.0
        elif speed == "1x":
            return 1.0
        elif speed == "2x":
            return 0.5
        else:
            return 1.0

    def build_prompt(self) -> str:
        return ("## Animation Description:"
                f"{self.state.animation_description}"
                "\n"
                "## Animation Duration:"
                f"{self.state.animation_duration} seconds")

    def generate_functions(self):
        prompt = self.build_prompt()

        status = self.gui_api.add_markdown("*Analyzing Prompt...*")
        progress = self.gui_api.add_progress_bar(value=0, animated=True)

        general_summary = prompt_4o(prompt=prompt,
                                    system_message=prompts.SUMMARY_SYS_MSG,
                                    response_format=AnimationSummary,
                                    temperature=self.state.temperature)
        self.state.animation_title = general_summary.__getattribute__("animation_title").title()
        self.state.centers_summary = extract_centers_summary(general_summary)
        self.state.rgbs_summary = extract_rgbs_summary(general_summary)
        self.state.opacities_summary = extract_opacities_summary(general_summary)

        progress.value = 25
        status.content = "*Generating Center Function...*"
        centers_generator_prompt = prompts.CENTERS_GENERATOR_TEMPLATE.format(centers_summary=self.state.centers_summary)
        centers_function_md = prompt_4o(prompt=centers_generator_prompt,
                                               system_message=prompts.GENERATOR_SYS_MSG,
                                               temperature=0.0)

        progress.value = 50
        status.content = "*Generating RGB Function...*"
        rgbs_generator_prompt = prompts.RGBS_GENERATOR_TEMPLATE.format(rgbs_summary=self.state.rgbs_summary)
        rgbs_function_md = prompt_4o(prompt=rgbs_generator_prompt,
                                            system_message=prompts.GENERATOR_SYS_MSG,
                                            temperature=0.0)
        progress.value = 75
        status.content = "*Generating Opacity Function...*"
        opacities_generator_prompt = prompts.OPACITIES_GENERATOR_TEMPLATE.format(opacities_summary=self.state.opacities_summary)
        opacities_function_md = prompt_4o(prompt=opacities_generator_prompt,
                                                 system_message=prompts.GENERATOR_SYS_MSG,
                                                 temperature=0.0)
        progress.value = 100

        centers_function_code = self.extract_python_code(centers_function_md)
        rgbs_function_code = self.extract_python_code(rgbs_function_md)
        opacities_function_code = self.extract_python_code(opacities_function_md)

        progress.remove()
        status.remove()

        return centers_function_code, rgbs_function_code, opacities_function_code


    def generate_functions_with_feedback(self, feedback: str):
        status = self.gui_api.add_markdown("*Implementing Feedback...*")
        progress = self.gui_api.add_progress_bar(value=0, animated=True)

        centers_system_msg = prompts.CENTERS_FEEDBACK.format(context=self.state.centers_context)
        centers_function_md = prompt_4o(prompt=feedback,
                                        system_message=centers_system_msg,
                                          temperature=0.0)
        progress.value = 15
        rgbs_system_msg = prompts.RGBS_FEEDBACK.format(context=self.state.rgbs_context)
        rgbs_function_md = prompt_4o(prompt=feedback,
                                     system_message=rgbs_system_msg,
                                          temperature=0.0)
        progress.value = 30
        opacities_system_msg = prompts.OPACITIES_FEEDBACK.format(context=self.state.opacities_context)
        opacities_function_md = prompt_4o(prompt=feedback,
                                          system_message=opacities_system_msg,
                                          temperature=0.0)
        progress.value = 45

        centers_function_code = self.extract_python_code(centers_function_md)
        rgbs_function_code = self.extract_python_code(rgbs_function_md)
        opacities_function_code = self.extract_python_code(opacities_function_md)
        
        exec(centers_function_code, globals())
        exec(rgbs_function_code, globals())
        exec(opacities_function_code, globals())

        progress.value = 100
        progress.remove()
        status.remove()

        return centers_function_code, rgbs_function_code, opacities_function_code