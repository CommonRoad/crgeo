from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np
T_Frame = Optional[np.ndarray]

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from crgeo.rendering.viewer.viewer_2d import Viewer2D

if TYPE_CHECKING:
    from crgeo.dataset.commonroad_data import CommonRoadData
    from crgeo.dataset.commonroad_data_temporal import CommonRoadDataTemporal
    from crgeo.simulation.base_simulation import BaseSimulation
    from crgeo.simulation.ego_simulation.ego_vehicle import EgoVehicle
    from crgeo.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
    from crgeo.rendering.traffic_scene_renderer import TrafficSceneRenderer


class SkipRenderInterrupt(BaseException):
    pass


@dataclass
class CameraView:
    position: np.ndarray # [x, y]
    orientation: float
    range: Optional[float] = None


@dataclass
class RenderParams:
    """
        RenderParams is the input data structure of renderer plugins:

        time_step(int): Count of numerical step of simulation. Simulation time is calculated by time_step * dt.
        scenario: Current scenario to be rendered
        planning_problem_set: Optional argument, contains the goal condition for ego vehicle
        ego_vehicle: Contains ego vehicle model information and ego_route
        ego_vehicle_simulation: Available if using EgoVehicleSimulation instead of ScenarioSimulation
        simulation: The simulation(including its simulation options) that the CommonRoadData generated from
        data: CommonRoadData instance at current time_step obtained by applying trafficextractor in simulation
        render_kwargs (Dict[str, Any]): Optional kwargs for TrafficSceneRenderer.

    """
    time_step: Optional[int] = None
    camera_view: Optional[CameraView] = None
    scenario: Optional[Scenario] = None
    planning_problem_set: Optional[PlanningProblemSet] = None
    ego_vehicle: Optional[EgoVehicle] = None
    ego_vehicle_simulation: Optional[EgoVehicleSimulation] = None
    simulation: Optional[BaseSimulation] = None
    data: Optional[Union[CommonRoadData, CommonRoadDataTemporal]] = None
    render_kwargs: Optional[Dict[str, Any]] = None


class Renderable(ABC):

    @abstractmethod
    def render(
        self,
        *,
        renderer: TrafficSceneRenderer,
        render_params: Optional[RenderParams] = None,
        return_rgb_array: bool = False,
        **render_kwargs: Any
    ) -> T_Frame:
        ...


T_RendererPlugin = Callable[[Viewer2D, RenderParams], None]
