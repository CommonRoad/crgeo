from typing import Dict, Tuple, Optional
import numpy as np
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.io_extensions.lanelet_network import collect_adjacent_lanelets

from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class DistanceToRoadBoundariesFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    """
    Returns the lateral distance from a vehicle to the (same-direction) left and right road boundaries.
    """
    def __init__(
        self,
    ) -> None:
        self._lanelet_id_to_road_boundaries: Dict[int, Tuple[ContinuousPolyline, ContinuousPolyline]] = {}
        self._last_scenario_id: Optional[str] = None
        super().__init__()

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        current_lanelet = simulation.get_obstacle_lanelet(params.obstacle)
        current_position = params.state.position

        if current_lanelet is None:
            return self._return_undefined_features()
        
        if current_lanelet.lanelet_id not in self._lanelet_id_to_road_boundaries:
            left_adjacent_lanelets, _ = collect_adjacent_lanelets(
                lanelet_network=simulation.lanelet_network,
                lanelet=current_lanelet,
                include_left=True,
                include_right=False
            )
            right_adjacent_lanelets, _ = collect_adjacent_lanelets(
                lanelet_network=simulation.lanelet_network,
                lanelet=current_lanelet,
                include_left=False,
                include_right=True
            )
            leftmost_lanelet = left_adjacent_lanelets[-1]
            rightmost_lanelet = right_adjacent_lanelets[-1]
            
            left_road_boundary = simulation.get_lanelet_left_polyline(leftmost_lanelet.lanelet_id)
            right_road_boundary = simulation.get_lanelet_right_polyline(rightmost_lanelet.lanelet_id)
            self._lanelet_id_to_road_boundaries[current_lanelet.lanelet_id] = (left_road_boundary, right_road_boundary)

        else:
            left_road_boundary, right_road_boundary = self._lanelet_id_to_road_boundaries[current_lanelet.lanelet_id]

        dist_left_road_boundary = left_road_boundary.get_lateral_distance(
            current_position,
            linear_projection=simulation.options.linear_lanelet_projection
        )
        dist_right_road_boundary = right_road_boundary.get_lateral_distance(
            current_position,
            linear_projection=simulation.options.linear_lanelet_projection
        )

        features: FeatureDict = {
            V_Feature.DistLeftRoadBound.value: dist_left_road_boundary,
            V_Feature.DistRightRoadBound.value: dist_right_road_boundary,
            V_Feature.IsOffroad.value: 0.0
        }

        return features
    
    def _return_undefined_features(self) -> FeatureDict:
        features: FeatureDict = {}
        features[V_Feature.DistLeftRoadBound.value] = np.nan
        features[V_Feature.DistRightRoadBound.value] = np.nan
        features[V_Feature.IsOffroad.value] = 1.0
        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        scenario_id = str(simulation.current_scenario.scenario_id)
        if scenario_id != self._last_scenario_id:
            self._lanelet_id_to_road_boundaries = {}
        self._last_scenario_id = scenario_id
