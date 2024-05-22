from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

if TYPE_CHECKING:
    from commonroad_geometric.dataset.commonroad_data import CommonRoadData
    from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
    from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
    from commonroad_geometric.simulation.base_simulation import BaseSimulation
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

T_Frame = np.ndarray
T_Position2D = Union[np.ndarray, Tuple[float, float]]
T_Position3D = Union[np.ndarray, Tuple[float, float, float]]
T_Angle = float
T_Vertices = np.ndarray


class SkipRenderInterrupt(BaseException):
    pass


@dataclass
class CameraView2D:
    center_position: np.ndarray  # [x, y]
    orientation: float
    view_range: Optional[float] = None


@dataclass
class RenderParams:
    """
    RenderParams is the input data structure of renderer plugins:

    Attributes:
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
    scenario: Optional[Scenario] = None
    planning_problem_set: Optional[PlanningProblemSet] = None
    ego_vehicle: Optional[EgoVehicle] = None
    ego_vehicle_simulation: Optional[EgoVehicleSimulation] = None
    simulation: Optional[BaseSimulation] = None
    data: Optional[Union[CommonRoadData, CommonRoadDataTemporal]] = None
    render_kwargs: Dict[str, Any] = field(default_factory=dict)


class Renderable(Protocol):
    """
    The Renderable protocol indicates that something can be rendered with a TrafficSceneRenderer.
    """

    def render(
        self,
        *,
        renderers: Sequence[TrafficSceneRenderer],
        render_params: Optional[RenderParams] = None,
        return_frames: bool = False,
        **render_kwargs: Dict[str, Any]
    ) -> Sequence[T_Frame]:
        ...
