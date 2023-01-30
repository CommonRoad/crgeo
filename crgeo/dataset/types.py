from typing import Union, Callable, Iterable, TypeVar, Any

from crgeo.dataset.commonroad_data import CommonRoadData
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

import torch_geometric.data
AnyData = Union[torch_geometric.data.Data, torch_geometric.data.HeteroData, CommonRoadData]
TransformCallable = Callable[[int, int, AnyData], Iterable[AnyData]]
T_IntermediateData = TypeVar("T_IntermediateData", bound=torch_geometric.data.data.BaseData)
T_Data = TypeVar("T_Data", bound=torch_geometric.data.data.BaseData)
T_PreTransform = Callable[[Scenario, PlanningProblemSet], Iterable[T_IntermediateData]]
T_PreFilter = Callable[[T_IntermediateData], bool]
T_Transform = Callable[[T_IntermediateData], T_Data]