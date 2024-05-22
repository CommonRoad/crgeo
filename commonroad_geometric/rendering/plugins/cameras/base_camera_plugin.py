from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from commonroad.scenario.scenario import Scenario

from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import CameraView2D, RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer

T_CameraPlugin = TypeVar("T_CameraPlugin", bound="BaseCameraPlugin")


class BaseCameraPlugin(BaseRenderPlugin, ABC):
    def __init__(
        self,
        fallback_camera: BaseCameraPlugin,
    ):
        super().__init__()
        self._fallback_camera = fallback_camera  # Can be self
        self._current_view: Optional[CameraView2D] = None
        self.__last_scenario: Optional[Scenario] = None

    @property
    def fallback_camera(self) -> BaseCameraPlugin:
        return self._fallback_camera

    @property
    def current_view(self) -> Optional[CameraView2D]:
        return self._current_view

    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        scenario = params.scenario
        # Remove this if we ever start modifying the lanelet network of a scenario during rendering
        is_new_scenario = self.__last_scenario is None or scenario.scenario_id != self.__last_scenario.scenario_id
        if is_new_scenario:
            self.__last_scenario = scenario
            # Clear the canvas of the viewer to get rid of leftover artifacts
            viewer.clear()
        # Delegate in case we need to change the signature of __call__ at some point
        self.set_camera(viewer, params)

    @abstractmethod
    def set_camera(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        ...
