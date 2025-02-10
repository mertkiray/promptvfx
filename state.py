from __future__ import annotations

from pathlib import Path
from splat_utils import SplatFile, load_splat
from src.viser._scene_handles import GaussianSplatHandle

class Observer:
    def update(self):
        pass

class Subject:
    def __init__(self):
        self._observers: list[Observer] = []

    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update()

class State(Subject):
    def __init__(self):
        super().__init__()
        self._splat_path: Path = Path("data/luigi.splat")
        self._fps: int = 8
        self._speed: str = "1x"
        self._splat: SplatFile = load_splat(self._splat_path)
        self.background: GaussianSplatHandle
        self._animation_name: str = "None"
        self.animation_description: str = ""
        self.temperature: float = 1.00
        self._animation_duration: int = 1
        self.centers_fn_md: str = "```\n```"
        self.rgbs_fn_md: str = "```\n```"
        self.opacities_fn_md: str = "```\n```"
        self.frame_to_gs_handle: dict[int, GaussianSplatHandle] = {}
        self.has_animation: bool = False
        self.centers_summary: str = ""
        self.rgbs_summary: str = ""
        self.opacities_summary: str = ""
        self.animation_running: bool = False

    @property
    def splat_path(self) -> Path:
        return self._splat_path
    
    @splat_path.setter
    def splat_path(self, value: Path) -> None:
        self._splat_path = value
        self._splat = load_splat(value)

    @property
    def animation_title(self) -> str:
        return self._animation_name

    @animation_title.setter
    def animation_title(self, value: str) -> None:
        self._animation_name = value
        self.notify()
    
    @property
    def animation_duration(self) -> int:
        return self._animation_duration
    
    @animation_duration.setter
    def animation_duration(self, value: int) -> None:
        self._animation_duration = value
        self.notify()

    @property
    def fps(self) -> int:
        return self._fps
    
    @fps.setter
    def fps(self, value: int) -> None:
        self._fps = value
        self.notify()
        
    @property
    def speed(self) -> str:
        return self._speed
    
    @speed.setter
    def speed(self, value: str) -> None:
        self._speed = value
        self.notify()

    @property
    def total_frames(self) -> int:
        return self.fps * self.animation_duration


    @property
    def centers_context(self) -> str:
        return f"""
Initial prompt:
{self.animation_description}

Initial LLM Analysis:
{self.centers_summary}

Initial Python Function:
{self.centers_fn_md}
"""

    @property
    def rgbs_context(self) -> str:
        return f"""
Initial prompt:
{self.animation_description}

Initial LLM Analysis:
{self.rgbs_summary}

Initial Python Function:
{self.rgbs_fn_md}
"""
    
    @property
    def opacities_context(self) -> str:
        return f"""
Initial prompt:
{self.animation_description}

Initial LLM Analysis:
{self.opacities_summary}

Initial Python Function:
{self.opacities_fn_md}
"""


    def change_to_frame(self, frame: int):
        for i, gs_handle in self.frame_to_gs_handle.items():
            if i == frame:
                gs_handle.visible = True
            else:
                gs_handle.visible = False


    def remove_gs_handles(self) -> None:
        for _, gs_handle in self.frame_to_gs_handle.items():
            if gs_handle:
                gs_handle.remove()
                del gs_handle
        self.frame_to_gs_handle = {}