from math import isfinite
from typing import Dict, Optional, Tuple

import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, L2LFeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import L2L_Feature
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType


class LaneletConnectionGeometryFeatureComputer(BaseFeatureComputer[L2LFeatureParams]):

    def __init__(
        self,
    ) -> None:
        self._feature_cache: Dict[Tuple[int, int], FeatureDict] = {}
        self._last_scenario_id: Optional[str] = None
        super().__init__()

    def __call__(
        self,
        params: L2LFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        source_id = params.source_lanelet.lanelet_id
        target_id = params.target_lanelet.lanelet_id

        if ((source_id, target_id)) in self._feature_cache:
            return self._feature_cache[(source_id, target_id)]

        source_lanelet_path = simulation.get_lanelet_center_polyline(source_id)
        target_lanelet_path = simulation.get_lanelet_center_polyline(target_id)

        edge_data = simulation.lanelet_graph.get_edge_data(source_id, target_id)
        if edge_data is None:
            features = {
                L2L_Feature.RelativeIntersectAngle.value: np.nan,
                L2L_Feature.RelativeStartAngle.value: np.nan,
                L2L_Feature.RelativeEndAngle.value: np.nan,
                L2L_Feature.SourcetArclengthAbs.value: np.nan,
                L2L_Feature.SourceArclengthRel.value: np.nan,
                L2L_Feature.TargetArclengthAbs.value: np.nan,
                L2L_Feature.TargetArclengthRel.value: np.nan,
                L2L_Feature.TargetCurvature.value: np.nan,
                L2L_Feature.SourceCurvature.value: np.nan,
                L2L_Feature.RelativeSourceLength.value: np.nan
            }
            self._feature_cache[(source_id, target_id)] = features
            return features

        # edge_type = LaneletEdgeType(edge_data['lanelet_edge_type'])
        # print(source_lanelet_path(edge_data['source_arclength']), target_lanelet_path(edge_data['target_arclength']))

        assert np.isfinite(source_lanelet_path(edge_data['source_arclength'])).all()
        assert np.isfinite(target_lanelet_path(edge_data['target_arclength'])).all()

        source_orientation = source_lanelet_path.get_direction(edge_data['source_arclength'])
        target_orientation = target_lanelet_path.get_direction(edge_data['target_arclength'])

        source_curvature = source_lanelet_path.get_curvature(edge_data['source_arclength'])
        target_curvature = target_lanelet_path.get_curvature(edge_data['target_arclength'])

        relative_intersect_angle = relative_orientation(source_orientation, target_orientation)
        relative_start_angle = relative_orientation(
            source_lanelet_path.get_direction(0),
            target_lanelet_path.get_direction(0)
        )
        relative_end_angle = relative_orientation(
            source_lanelet_path.get_direction(source_lanelet_path.length),
            target_lanelet_path.get_direction(target_lanelet_path.length)
        )
        assert isfinite(relative_intersect_angle)
        assert isfinite(relative_start_angle)
        assert isfinite(relative_end_angle)

        features = {
            L2L_Feature.RelativeIntersectAngle.value: relative_intersect_angle,
            L2L_Feature.RelativeStartAngle.value: relative_start_angle,
            L2L_Feature.RelativeEndAngle.value: relative_end_angle,
            L2L_Feature.SourcetArclengthAbs.value: edge_data['source_arclength'],
            L2L_Feature.TargetArclengthAbs.value: edge_data['target_arclength'],
            L2L_Feature.SourceArclengthRel.value: edge_data['source_arclength_rel'],
            L2L_Feature.TargetArclengthRel.value: edge_data['target_arclength_rel'],
            L2L_Feature.SourceCurvature.value: source_curvature,
            L2L_Feature.TargetCurvature.value: target_curvature,
            L2L_Feature.RelativeSourceLength.value: source_lanelet_path.length / target_lanelet_path.length
        }

        self._feature_cache[(source_id, target_id)] = features

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        scenario_id = str(simulation.current_scenario.scenario_id)
        if scenario_id != self._last_scenario_id:
            self._feature_cache = {}
        self._last_scenario_id = scenario_id
