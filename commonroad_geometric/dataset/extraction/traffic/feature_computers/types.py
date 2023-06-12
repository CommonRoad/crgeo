from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING, TypeVar, Union

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State
from torch import Tensor
from typing_extensions import TypeAlias

from commonroad_geometric.simulation.base_simulation import BaseSimulation

if TYPE_CHECKING:
    from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
    from commonroad_geometric.simulation.ego_simulation.planning_problem.ego_route import EgoRoute
    from commonroad_geometric.dataset.extraction.traffic.feature_computers import BaseFeatureComputer

FeatureValue: TypeAlias = Union[int, bool, float, Tensor]
FeatureDict: TypeAlias = Dict[str, FeatureValue]


@dataclass
class BaseFeatureParams:
    """
    dt (float): Positive time delta between each time-step. Integral granularity, numerical step-size of simulation.
    time_step(int): Count of numerical step of simulation. Simulation time is calculated by time_step * dt.
    """
    dt: float
    time_step: int


@dataclass
class VFeatureParams(BaseFeatureParams):
    """
    obstacle: Each dynamic obstacle has stored its predicted movement in future time steps.
    state: consult State class definition
    is_ego_vehicle: a boolean mask, True if the vehicle is ego vehicle
    ego_route: High-level interface for route/trajectory planning problem of EgoVehicle in EgoVehicleSimulation
    """
    obstacle: DynamicObstacle
    state: State
    is_ego_vehicle: bool
    ego_state: Optional[State]
    ego_route: Optional[EgoRoute]


@dataclass
class V2VFeatureParams(BaseFeatureParams):
    """
    data structure describing vehicle to vehicle edge feature
    distance: euclidean distance between source and target vehicle
    source->target: mark the edge direction if using directed graph
    """
    distance: float
    source_obstacle: DynamicObstacle
    source_state: State
    source_is_ego_vehicle: bool
    target_obstacle: DynamicObstacle
    target_state: State
    target_is_ego_vehicle: bool
    ego_state: Optional[State]


@dataclass
class LFeatureParams(BaseFeatureParams):
    lanelet: Lanelet
    ego_state: Optional[State]


@dataclass
class L2LFeatureParams(BaseFeatureParams):
    source_lanelet: Lanelet
    target_lanelet: Lanelet
    ego_state: Optional[State]


@dataclass
class V2LFeatureParams(BaseFeatureParams):
    obstacle: DynamicObstacle
    state: State
    is_ego_vehicle: bool
    lanelet: Lanelet
    ego_state: Optional[State]


@dataclass
class VTVFeatureParams(BaseFeatureParams):
    data: CommonRoadDataTemporal
    past_obstacle_idx: int
    curr_obstacle_idx: int


T_FeatureParams = TypeVar("T_FeatureParams", bound=BaseFeatureParams)


T_FeatureCallable: TypeAlias = Union[
    Callable[[T_FeatureParams, BaseSimulation], FeatureDict],
    Callable[[T_FeatureParams], FeatureDict],
]


T_FeatureComputer: TypeAlias = Union[
    "BaseFeatureComputer[T_FeatureParams]",
    T_FeatureCallable[T_FeatureParams],
]
