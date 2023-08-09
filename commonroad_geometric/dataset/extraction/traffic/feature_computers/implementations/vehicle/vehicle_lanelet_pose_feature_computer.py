from typing import Dict, Set

import numpy as np

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import VehicleLaneletPoseEdgeFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


# TODO: Dupliacte with v2l implementation
class VehicleLaneletPoseFeatureComputer(VehicleLaneletPoseEdgeFeatureComputer):
    """
    In comparison with VehicleLaneletPoseEdgeFeatureComputer for V2L feature extraction,
    VehicleLaneletPoseFeatureComputer provides the most likely lanelet association of each vehicle,
    while VehicleLaneletPoseEdgeFeatureComputer provides all lanelet association when vehicle is on multiple lanelets. 
    """
    @classproperty
    def skip_normalize_features(cls) -> Set[str]:
        return {V_Feature.HeadingError.value}


    def __init__(
        self,
        include_longitudinal_abs: bool = True,
        include_longitudinal_rel: bool = True,
        include_lateral_left: bool = True,
        include_lateral_right: bool = True,
        include_lateral_error: bool = True,
        include_heading_error: bool = True,
        update_exact_interval: int = 1,
    ) -> None:
        if not any((
            include_longitudinal_abs,
            include_longitudinal_rel,
            include_lateral_left,
            include_lateral_right,
            include_lateral_error,
            include_heading_error
        )):
            raise ValueError("VehicleLaneletPoseFeatureComputer doesn't include any features")

        super().__init__(
            include_longitudinal_abs=include_longitudinal_abs,
            include_longitudinal_rel=include_longitudinal_rel,
            include_lateral_left=include_lateral_left,
            include_lateral_right=include_lateral_right,
            include_lateral_error=include_lateral_error,
            include_heading_error=include_heading_error
        )

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        features=self.compute_features(
            params,
            lanelet=simulation.get_obstacle_lanelet(params.obstacle),
            simulation=simulation
        )
            
        return features

    def _return_undefined_features(self) -> Dict[str, float]:
        features: Dict[str, float] = {}

        if self._include_longitudinal_abs:
            features[V_Feature.LaneletArclengthAbs.value] = np.nan
        if self._include_longitudinal_rel:
            features[V_Feature.LaneletArclengthRel.value] = np.nan
        if self._include_lateral_left:
            features[V_Feature.DistLeftBound.value] = np.nan
        if self._include_lateral_right:
            features[V_Feature.DistRightBound.value] = np.nan
        if self._include_lateral_error:
            features[V_Feature.LaneletLateralError.value] = np.nan
        if self._include_heading_error:
            features[V_Feature.HeadingError.value] = np.nan

        return features





