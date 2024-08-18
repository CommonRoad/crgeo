from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Sequence, Union
import logging
from copy import deepcopy

from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.extraction.base_extractor import BaseExtractor, BaseExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VTVFeatureParams
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractionParams
from commonroad_geometric.simulation.base_simulation import Unlimited
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor


logger = logging.getLogger(__name__)

@dataclass
class _TimestampData:
    time_step: int
    data: CommonRoadData = field(default_factory=lambda: CommonRoadData())

    def __post_init__(self):
        self.obstacle_id_to_obstacle_idx = {v.item(): k for k, v in enumerate(self.data.v.id)}



@dataclass
class TemporalTrafficExtractorOptions(BaseExtractorOptions):
    r"""
    Configuration for TemporalTrafficExtractor.

    Attributes:
        collect_num_time_steps (int): Number of (past) time steps to merge into a temporal graph.
        collect_skip_time_steps (int): Number of time steps to skip after returning a complete temporal graph.
        return_incomplete_temporal_graph (bool): Whether to return a temporal graph if less than collect_num_time_steps
                                                 time steps are. Defaults to False.
        combine_time_steps (bool): Whether to return a list of graphs (one for each time step) or a combined temporal
                                   graph. Defaults to True.
        add_temporal_vehicle_edges (bool): Whether to add temporal vehicle-to-vehicle edges. Defaults to True.
        max_time_steps_temporal_edge (T_CountParam): If not Unlimited, only create temporal edges between vehicle nodes
                                                     which are at most max_time_steps_temporal_edge time steps away
                                                     from each other. Defaults to Unlimited.
        temporal_vehicle_edge_feature_computers (Optional[Sequence[Callable[[VTVFeatureParams], List[float]]]]):
            Optional feature computers for temporal edges.
    """
    collect_num_time_steps: int = 10
    collect_skip_time_steps: int = 0
    no_skip: bool = False
    return_incomplete_temporal_graph: bool = False
    combine_time_steps: bool = True
    add_temporal_vehicle_edges: bool = True
    max_time_steps_temporal_edge: T_CountParam = Unlimited
    temporal_vehicle_edge_feature_computers: Optional[Sequence[Callable[[VTVFeatureParams], List[float]]]] = None
    postprocessors: Sequence[T_LikeBaseDataPostprocessor] = field(default_factory=list)
    disable_postprocessing: bool = False


ReturnType = Union[None, Sequence[CommonRoadData], CommonRoadDataTemporal]


class TemporalTrafficExtractor(BaseExtractor[TemporalTrafficExtractorOptions, TrafficExtractionParams, ReturnType]):

    def __init__(
        self,
        traffic_extractor: TrafficExtractor,
        options: TemporalTrafficExtractorOptions,
    ) -> None:
        super().__init__(simulation=traffic_extractor.simulation, options=options)
        self._traffic_extractor = traffic_extractor
        self._past_time_steps: Deque[_TimestampData] = deque(maxlen=self.options.collect_num_time_steps)
        self._skip_steps: int = 0

    def extract(
        self,
        time_step: int,
        params: TrafficExtractionParams
    ) -> ReturnType:
        r"""
        Extracts a heterogeneous spatio-temporal, graph representation of vehicles and lanelets in a CommonRoad scenario
        intended for PyTorch geometric. I.e. combines several CommonRoadData instances across time.

        Args:
            time_step (int): Time step for which next data instance should be extracted.
            params (T_BaseExtractionParams): Additional parameters for extracting the data instance.

        Returns:
            Union[None, Sequence[CommonRoadData], CommonRoadDataTemporal]:
             None - If, return_incomplete_temporal_graph is set to False or options no_skip is True.
             Sequence of Pytorch-Geometric CommonRoadData instance representing the scene.
        """
        data = self._traffic_extractor.extract(time_step=time_step, params=params)

        self._past_time_steps.append(_TimestampData(
            time_step=time_step,
            data=data
        ))

        if not self.options.no_skip and self._skip_steps > 0:
            self._skip_steps -= 1
            return None

        if (len(self._past_time_steps) < self.options.collect_num_time_steps and
            not self.options.return_incomplete_temporal_graph):
            return None

        elif len(self._past_time_steps) == self.options.collect_num_time_steps:
            self._skip_steps = self.options.collect_skip_time_steps

        if not self.options.combine_time_steps:
            return [t.data for t in self._past_time_steps]

        # Combine _past_time_steps into one graph
        temporal_data = CommonRoadDataTemporal.from_data_list(
            data_list=[ts.data for ts in self._past_time_steps],
            delta_time=self._simulation.dt,
        )
        if self.options.add_temporal_vehicle_edges:
            # Add temporal vehicle-to-vehicle edges
            CommonRoadDataTemporal.add_temporal_vehicle_edges_(
                data=temporal_data,
                max_time_steps_temporal_edge=self.options.max_time_steps_temporal_edge,
                obstacle_id_to_obstacle_idx=[ts.obstacle_id_to_obstacle_idx for ts in self._past_time_steps],
                feature_computers=self.options.temporal_vehicle_edge_feature_computers,
            )

        if temporal_data.vtv.num_edges > 0:
            assert temporal_data.vtv.edge_index.max() < temporal_data.v.num_nodes

        if not params.disable_postprocessing:
            for postprocessor in self.options.postprocessors:
                result = postprocessor(
                    [temporal_data],
                    simulation=self.simulation,
                    ego_vehicle=params.ego_vehicle
                )
                if len(result) == 1:
                    temporal_data = result[0]
                else:
                    logger.warning(f"Skipping ambiguous postprocessing routine {type(postprocessor).__name__} "
                                   f"(returned {len(result)} samples)")


        return temporal_data

    def reset_feature_computers(self) -> None:
        self._traffic_extractor.reset_feature_computers()

    def __iter__(self) -> BaseExtractor:
        return self._traffic_extractor.__iter__()

    def __next__(self) -> CommonRoadData:
        return self._traffic_extractor.__next__()

    def __len__(self) -> T_CountParam:
        return self._traffic_extractor.__len__()
