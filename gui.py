import re
import time
import traceback
import numpy as np
from state import State, Observer
import viser
from viser._gui_api import GuiApi
from viser._scene_api import SceneApi
from viser._viser import ViserServer
from llm_utils import prompt_4o,  prompt_o1
import prompts


class Gui(Observer):
    def __init__(self, server: ViserServer, state: State):
        self.scene_api: SceneApi = server.scene
        self.gui_api: GuiApi = server.gui
        self.gui_api.configure_theme(dark_mode=True)
        self.state: State = state
        self.state.frame_to_gs_handle[0] = self.add_splat_at_t(0)

        self.title_md = self.gui_api.add_markdown(f"Animation: **{self.state.animation_title}**")

        self.prompt_btn = self.gui_api.add_button("Prompt", icon=viser.Icon.ROBOT)
        @self.prompt_btn.on_click
        def _(_) -> None:
            self.open_prompt()

        self.inspect_btn = self.gui_api.add_button("Inspect Functions", icon=viser.Icon.FUNCTION)
        @self.inspect_btn.on_click
        def _(_) -> None:
            self.open_inspector()

        self.reload_btn = server.gui.add_button("Reload Frames", icon=viser.Icon.RESTORE)
        @self.reload_btn.on_click
        def _(_) -> None:
            self.reload_splats()
        
        self.fps_dropdown = self.gui_api.add_dropdown(
            label="FPS",
            options=["8", "24", "60", "144"],
            initial_value="24"
        )
        @self.fps_dropdown.on_update
        def _(_) -> None:
            self.state.fps = int(self.fps_dropdown.value)
            if self.check_functions_defined():
                self.reload_splats()


        self.speed_dropdown = self.gui_api.add_dropdown(
            label="Speed",
            options=["0.25x", "0.5x", "1x", "2x"],
            initial_value="1x"
        )

        self.frame_slider = self.gui_api.add_slider(
            label=f"Current Frame",
            min=0,
            max=self.state.total_frames-1,
            step=1,
            initial_value=0,
        )
        @self.frame_slider.on_update
        def _(_) -> None:
            self.state.change_to_frame(self.frame_slider.value)

        self.play_btn = server.gui.add_button("▶ Play", color="green")
        @self.play_btn.on_click
        def _(_) -> None:
            self.play_btn.disabled = True
            self.fps_dropdown.disabled = True
            self.play_animation()
            self.play_btn.disabled = False
            self.fps_dropdown.disabled = False

    def update(self):
        self.title_md.content = f"Animation: **{self.state.animation_title}**"
        self.frame_slider.max = self.state.total_frames-1


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
            f"splat_0",
            centers=self.state.splat["centers"],
            rgbs=self.state.splat["rgbs"],
            opacities=self.state.splat["opacities"],
            covariances=self.state.splat["covariances"],
            )
        
        try:
            centers = self.state.splat["centers"]
            rgbs = self.state.splat["rgbs"]
            opacities = self.state.splat["opacities"]
            centers_at_t = globals()["compute_centers"](t, centers)
            rbgs_at_t = globals()["compute_rgbs"](t, rgbs)
            opacities_at_t = globals()["compute_opacities"](t, opacities)
            return self.scene_api.add_gaussian_splats(
                f"splat_{t}",
                centers=centers_at_t,
                rgbs=rbgs_at_t,
                opacities=opacities_at_t,
                covariances=self.state.splat["covariances"],
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
        self.state.change_to_frame(self.frame_slider.value)
        progress_bar.remove()
        loading_md.remove()

    def reload_splats(self) -> None:
        if not self.check_functions_defined():
            with self.gui_api.add_modal("⚠ Please generate functions first!") as popout:
                close_btn = self.gui_api.add_button("X", color="red")
                @close_btn.on_click
                def _(_) -> None:
                    popout.close()
                raise Exception()
        
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
            title_txt = self.gui_api.add_text("Title", self.state.animation_title)
            description_txt = self.gui_api.add_text("Description", self.state.animation_description)
            temperatur_slider = self.gui_api.add_slider(
                label="Temperature",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=self.state.temperature
            )
            duration_slider = self.gui_api.add_slider(
                label="Duration (s)",
                min=1,
                max=5,
                step=1,
                initial_value=self.state.animation_duration
            )

            generate_btn = self.gui_api.add_button("🛠 Generate")
            @generate_btn.on_click
            def _(_) -> None:
                if not description_txt.value:
                    return
                popout.close()

                self.state.animation_title = title_txt.value
                self.state.animation_description = description_txt.value
                self.state.temperature = temperatur_slider.value
                self.state.animation_duration = duration_slider.value

                prompt = self.build_prompt()
                centers_fn, rgbs_fn, opacities_fn = self.generate_functions(prompt)
                exec(centers_fn, globals())
                exec(rgbs_fn, globals())
                exec(opacities_fn, globals())
                self.reload_splats()

                self.state.centers_fn_md = self.as_code_block(centers_fn)
                self.state.rgbs_fn_md = self.as_code_block(rgbs_fn)
                self.state.opacities_fn_md = self.as_code_block(opacities_fn)
                self.state.has_animation = True

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
                self.state.remove_gs_handles()
                self.load_splats()

            
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
            self.gui_api.add_markdown(self.state.centers_fn_md)
            self.gui_api.add_markdown(self.state.rgbs_fn_md)
            self.gui_api.add_markdown(self.state.opacities_fn_md)

            close_btn = self.gui_api.add_button("⨯", color="red")
            @close_btn.on_click
            def _(_) -> None:
                popout.close()

    def play_animation(self):
        animation_stopped = False

        stop_btn = self.gui_api.add_button("◼ Stop", color="red")
        @stop_btn.on_click
        def _(_) -> None:
            nonlocal animation_stopped
            animation_stopped=True
            stop_btn.remove()

        while True:
            for frame in range(self.frame_slider.value+1, self.state.total_frames):
                if animation_stopped:
                    return
                self.state.change_to_frame(frame)
                self.frame_slider.value = frame
                time.sleep((1.0/self.state.fps) * self.speed_to_sleep_factor(self.speed_dropdown.value))
            self.state.change_to_frame(0)
            self.frame_slider.value = 0
            time.sleep((1.0/self.state.fps) * self.speed_to_sleep_factor(self.speed_dropdown.value))


    def check_functions_defined(self) -> bool:
        fn_names = ["compute_centers", "compute_rgbs", "compute_opacities"]
        return all(name in globals() for name in fn_names)


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
        return ("## Animation Title:"
                f"{self.state.animation_title}"
                "\n"
                "## Animation Description:"
                f"{self.state.animation_description}"
                "\n"
                "## Animation Duration (in seconds):"
                f"{self.state.animation_duration}")

    def generate_functions(self, prompt: str):
        status = self.gui_api.add_markdown("*Analyzing Prompt...*")
        progress = self.gui_api.add_progress_bar(value=0, animated=True)

        general_summary = prompt_4o(prompt=prompt, system_message=prompts.GENERAL_SUMMARY_SYS_MSG, temperature=self.state.temperature)
        self.state.general_summary = general_summary
        progress.value = 10
        centers_summary = prompt_4o(prompt=general_summary, system_message=prompts.CENTERS_SUMMARY_SYS_MSG, temperature=self.state.temperature)
        self.state.centers_summary = centers_summary
        progress.value = 20
        rgbs_summary = prompt_4o(prompt=general_summary, system_message=prompts.RGBS_SUMMARY_SYS_MSG, temperature=self.state.temperature)
        self.state.rgbs_summary = rgbs_summary
        progress.value = 30
        opacities_summary = prompt_4o(prompt=general_summary, system_message=prompts.OPACITIES_SUMMARY_SYS_MSG, temperature=self.state.temperature)
        self.state.opacities_summary = opacities_summary
        progress.value = 40

        status.content = "*Generating Functions...*"
        # raw_centers_function = prompt_4o(prompt=centers_summary, system_message=prompts.CENTERS_GENERATOR_TEMPLATE, temperature=self.state.temperature)
        # progress.value = 40
        # raw_rgbs_function = prompt_4o(prompt=rgbs_summary, system_message=prompts.RGBS_GENERATOR, temperature=self.state.temperature)
        # progress.value = 50
        # raw_opacities_function = prompt_4o(prompt=opacities_summary, system_message=prompts.OPACITIES_GENERATOR, temperature=self.state.temperature)
        # progress.value = 60

        # status.content = "*Validating Python...*"
        validated_centers_function = prompt_o1(prompt=prompts.CENTERS_GENERATOR_TEMPLATE.format(centers_summary=centers_summary))
        progress.value = 60
        validated_rgbs_function = prompt_o1(prompt=prompts.RGBS_GENERATOR_TEMPLATE.format(rgbs_summary=rgbs_summary))
        progress.value = 80
        validated_opacities_function = prompt_o1(prompt=prompts.OPACITIES_GENERATOR_TEMPLATE.format(opacities_summary=opacities_summary))
        progress.value = 100

        executable_centers_function = self.extract_python_code(validated_centers_function)
        executable_rgbs_function = self.extract_python_code(validated_rgbs_function)
        executable_opacities_function = self.extract_python_code(validated_opacities_function)

        progress.remove()
        status.remove()

        return executable_centers_function, executable_rgbs_function, executable_opacities_function


    def generate_functions_with_feedback(self, feedback: str):
        status = self.gui_api.add_markdown("*Implementing Feedback...*")
        progress = self.gui_api.add_progress_bar(value=0, animated=True)

        centers_system_msg = prompts.CENTERS_FEEDBACK.format(context=self.state.centers_context)
        raw_centers_function = prompt_4o(prompt=feedback, system_message=centers_system_msg, temperature=self.state.temperature)
        progress.value = 15
        rgbs_system_msg = prompts.RGBS_FEEDBACK.format(context=self.state.rgbs_context)
        raw_rgbs_function = prompt_4o(prompt=feedback, system_message=rgbs_system_msg, temperature=self.state.temperature)
        progress.value = 30
        opacities_system_msg = prompts.OPACITIES_FEEDBACK.format(context=self.state.opacities_context)
        raw_opacities_function = prompt_4o(prompt=feedback, system_message=opacities_system_msg, temperature=self.state.temperature)
        progress.value = 45

        status.content = "*Validating Python...*"
        validated_centers_function = prompt_4o(prompt=raw_centers_function, system_message=prompts.PYTHON_VALIDATION, temperature=self.state.temperature)
        progress.value = 60
        validated_rgbs_function = prompt_4o(prompt=raw_rgbs_function, system_message=prompts.PYTHON_VALIDATION, temperature=self.state.temperature)
        progress.value = 75
        validated_opacities_function = prompt_4o(prompt=raw_opacities_function, system_message=prompts.PYTHON_VALIDATION, temperature=self.state.temperature)
        progress.value = 90

        executable_centers_function = self.extract_python_code(validated_centers_function)
        executable_rgbs_function = self.extract_python_code(validated_rgbs_function)
        executable_opacities_function = self.extract_python_code(validated_opacities_function)
        
        exec(executable_centers_function, globals())
        exec(executable_rgbs_function, globals())
        exec(executable_opacities_function, globals())

        progress.value = 100
        progress.remove()
        status.remove()

        return executable_centers_function, executable_rgbs_function, executable_opacities_function


