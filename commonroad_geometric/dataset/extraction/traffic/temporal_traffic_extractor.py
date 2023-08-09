from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Sequence, TYPE_CHECKING, Union

from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.extraction import BaseExtractor, BaseExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams
from commonroad_geometric.simulation.base_simulation import Unlimited

if TYPE_CHECKING:
    from commonroad_geometric.dataset.extraction.traffic import TrafficExtractor
    from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VTVFeatureParams


class _TimestampData(NamedTuple):
    time_step: int
    data: CommonRoadData
    obstacle_id_to_obstacle_idx: Dict[int, int]


@dataclass
class TemporalTrafficExtractorOptions(BaseExtractorOptions):
    """Configuration for TemporalTrafficExtractor

    - collect_num_time_steps: Number of (past) time steps to merge into a temporal graph.
    - collect_skip_time_steps: Number of time steps to skip after returning a complete temporal graph.
    - return_incomplete_graph: Whether to return a temporal graph if less than collect_num_time_steps time steps are
      available. If True an incomplete temporal graph with all currently available time steps is returned. If False
      None is returned.
    - combine_time_steps: Whether to return a list of graphs (one for each time step) or a combined temporal graph.
    - add_temporal_vehicle_edges: Whether to add temporal vehicle-to-vehicle edges.
    - max_time_steps_temporal_edge: If not Unlimited, only create temporal edges between vehicle nodes which are at most
      max_time_steps_temporal_edge time steps away from each other.
    """
    collect_num_time_steps: int
    collect_skip_time_steps: int = 0
    return_incomplete_temporal_graph: bool = False
    combine_time_steps: bool = True
    add_temporal_vehicle_edges: bool = True
    max_time_steps_temporal_edge: T_CountParam = Unlimited
    temporal_vehicle_edge_feature_computers: Optional[Sequence[Callable[[VTVFeatureParams], List[float]]]] = None


class TemporalTrafficExtractor(BaseExtractor[TrafficExtractionParams, Union[None, Sequence[CommonRoadData], CommonRoadDataTemporal]]):

    def __init__(
        self,
        traffic_extractor: TrafficExtractor,
        options: TemporalTrafficExtractorOptions,
    ) -> None:
        super().__init__(simulation=traffic_extractor.simulation, options=options)
        self._traffic_extractor = traffic_extractor

        self._collect_num_time_steps = options.collect_num_time_steps
        self._collect_skip_time_steps = options.collect_skip_time_steps
        self._return_incomplete_temporal_graph = options.return_incomplete_temporal_graph
        self._combine_time_steps = options.combine_time_steps
        self._add_temporal_vehicle_edges = options.add_temporal_vehicle_edges
        self._max_time_steps_temporal_edge = options.max_time_steps_temporal_edge
        self._temporal_vehicle_edge_feature_computers = options.temporal_vehicle_edge_feature_computers

        self._past_time_steps: Deque[_TimestampData] = deque(maxlen=self._collect_num_time_steps)
        self._skip_steps: int = 0

    def extract(
        self,
        params: TrafficExtractionParams
    ) -> Union[None, Sequence[CommonRoadData], CommonRoadDataTemporal]:

        # Domain-ambiguous term index from BaseExtractor refers to a time_step here
        time_step = params.index

        data = self._traffic_extractor.extract(params)

        self._past_time_steps.append(_TimestampData(
            time_step=time_step,
            data=data,
            obstacle_id_to_obstacle_idx=self._simulation.obstacle_id_to_obstacle_idx,
        ))

        if not params.no_skip and self._skip_steps > 0:
            self._skip_steps -= 1
            return None

        if len(self._past_time_steps) < self._collect_num_time_steps and not self._return_incomplete_temporal_graph:
            return None

        elif len(self._past_time_steps) == self._collect_num_time_steps:
            self._skip_steps = self._collect_skip_time_steps

        if not self._combine_time_steps:
            return [t.data for t in self._past_time_steps]

        # combine _past_time_steps into one graph
        data = CommonRoadDataTemporal.from_data_list(
            data_list=[ts.data for ts in self._past_time_steps],
            delta_time=self._simulation.dt,
        )
        if self._add_temporal_vehicle_edges:
            # add temporal vehicle-to-vehicle edges
            CommonRoadDataTemporal.add_temporal_vehicle_edges_(
                data=data,
                max_time_steps_temporal_edge=self._max_time_steps_temporal_edge,
                obstacle_id_to_obstacle_idx=[ts.obstacle_id_to_obstacle_idx for ts in self._past_time_steps],
                feature_computers=self._temporal_vehicle_edge_feature_computers,
            )
        return data

    def close(self) -> None:
        self._traffic_extractor.close()

    def reset_feature_computers(self) -> None:
        self._traffic_extractor.reset_feature_computers()

    def __iter__(self) -> BaseExtractor:
        return self._traffic_extractor.__iter__()

    def __next__(self) -> CommonRoadData:
        return self._traffic_extractor.__next__()

    def __len__(self) -> T_CountParam:
        return self._traffic_extractor.__len__()
