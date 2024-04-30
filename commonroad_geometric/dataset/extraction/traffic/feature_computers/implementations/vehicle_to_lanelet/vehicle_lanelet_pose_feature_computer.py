from math import cos
from typing import Dict, Set

import numpy as np
from commonroad.scenario.lanelet import Lanelet

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2L_Feature, V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, V2LFeatureParams, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class VehicleLaneletPoseEdgeFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    """

    Absolute Lanelet Arclength: The absolute arclength between the vehicle's cloest point on its lanelet center continuous polyline 
                                and the start point of lanelet center continuous polyline.
    Relative Lanelet Arclength: The relative lanelet arclength of the vehicle is the absolute arclength normalized 
                                using the length of the lanelet.
    Dist Left/Right Bound:  The absolute distance between a vehicle and the continuous polyline of the left or right boundary of 
                            a particular lanelet by finding the point closest to the vehicle located on the center polyline of the lanelet.
    Lateral Error:  The closest distance between a vehicle and the lanelet center continuous polyline
                    (positive when vehicle is on the right of center line)
    Heading Error: The orientation error between the continuous polyline of the lanelet and the orientation of the vehicle.
    
    """
    @classproperty
    def skip_normalize_features(cls) -> Set[str]:
        return {V2L_Feature.HeadingError.value}

    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(
        self,
        include_longitudinal_abs: bool = True,
        include_longitudinal_rel: bool = True,
        include_lateral_left: bool = True,
        include_lateral_right: bool = True,
        include_lateral_error: bool = True,
        include_heading_error: bool = True,
        update_exact_interval: int = 1,
        allow_outside_arclengths: bool = True,
        nan_if_missing: bool = False,
        linear_lanelet_projection: bool = False
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

        self._include_longitudinal_abs = include_longitudinal_abs
        self._include_longitudinal_rel = include_longitudinal_rel
        self._include_lateral_left = include_lateral_left
        self._include_lateral_right = include_lateral_right
        self._include_lateral_error = include_lateral_error
        self._include_heading_error = include_heading_error
        self._allow_outside_arclengths = allow_outside_arclengths
        self._linear_lanelet_projection = linear_lanelet_projection
        self._update_exact_interval = update_exact_interval
        self._use_approximations = update_exact_interval > 1
        self._nan_if_missing = nan_if_missing
        self._undefined_features = self._return_undefined_features()
        self._vehicle_call_count: Dict[int, int] = {}
        self._vehicle_arclength_approximations: Dict[int, FeatureDict] = {}
        super().__init__()

    def compute_features(
        self,
        params: V2LFeatureParams,
        lanelet: Lanelet,
        simulation: BaseSimulation
    ):
        if not hasattr(self, '_allow_outside_arclengths'): # TODO: remove
            self._allow_outside_arclengths = True

        _features: FeatureDict = {}
        if lanelet is None:
            return self._undefined_features
        obstacle_id = params.obstacle.obstacle_id
        if self._use_approximations and obstacle_id not in self._vehicle_call_count:
            self._vehicle_call_count[obstacle_id] = 0

        center_polyline = simulation.get_lanelet_center_polyline(lanelet.lanelet_id)
        if not self._use_approximations or \
            self._vehicle_call_count[obstacle_id] % self._update_exact_interval == 0 or \
            simulation.has_changed_lanelet(params.obstacle) or \
            obstacle_id not in self._vehicle_arclength_approximations or \
            params.time_step not in self._vehicle_arclength_approximations[obstacle_id]:
            lanelet_arclength_abs = center_polyline.get_projected_arclength(
                params.state.position,
                relative=False,
                linear_projection=self._linear_lanelet_projection
            )
            if self._allow_outside_arclengths and lanelet_arclength_abs <= 0:
                lanelet_arclength_abs = center_polyline.get_projected_arclength(
                    params.state.position + np.array(
                        np.cos(params.state.orientation) * params.obstacle.obstacle_shape.length / 2,
                        np.sin(params.state.orientation) * params.obstacle.obstacle_shape.length / 2
                    ),
                    relative=False,
                    linear_projection=self._linear_lanelet_projection
                ) - params.obstacle.obstacle_shape.length / 2
            elif self._allow_outside_arclengths and lanelet_arclength_abs >= center_polyline.length:
                lanelet_arclength_abs = center_polyline.get_projected_arclength(
                    params.state.position + np.array(
                        np.cos(-params.state.orientation) * params.obstacle.obstacle_shape.length / 2,
                        np.sin(-params.state.orientation) * params.obstacle.obstacle_shape.length / 2
                    ),
                    relative=False,
                    linear_projection=self._linear_lanelet_projection
                ) + params.obstacle.obstacle_shape.length / 2
        else:
            lanelet_arclength_abs = self._vehicle_arclength_approximations[obstacle_id][params.time_step]
        _features[V_Feature.LaneletArclengthAbs.value] = lanelet_arclength_abs
        if self._use_approximations:
            if obstacle_id not in self._vehicle_call_count:
                self._vehicle_call_count[obstacle_id] = 0 # not sure why it's necessary, avoids some KeyError
            self._vehicle_call_count[obstacle_id] += 1
        if self._include_longitudinal_rel:
            lanelet_arclength_rel = lanelet_arclength_abs / center_polyline.length
            _features[V_Feature.LaneletArclengthRel.value] = lanelet_arclength_rel
        if self._include_lateral_left or self._include_lateral_error:
            left_polyline = simulation.get_lanelet_left_polyline(lanelet.lanelet_id)
            dist_left_bound = left_polyline.get_projected_distance(
                params.state.position, 
                arclength=lanelet_arclength_abs
            )
            _features[V_Feature.DistLeftBound.value]=dist_left_bound
        if self._include_lateral_right or self._include_lateral_error:
            right_polyline = simulation.get_lanelet_right_polyline(lanelet.lanelet_id)
            dist_right_bound = right_polyline.get_projected_distance(
                params.state.position,
                arclength=lanelet_arclength_abs
            )
            _features[V_Feature.DistRightBound.value]=dist_right_bound

        if self._include_lateral_error:
            lateral_error = (dist_left_bound - dist_right_bound) / 2
            # same as  lateral_error = dist_left_bound - (dist_right_bound + dist_left_bound) / 2
            _features[V_Feature.LaneletLateralError.value] = lateral_error
        if self._include_heading_error:
            heading_error = relative_orientation(
                params.state.orientation,
                center_polyline.get_direction(lanelet_arclength_abs)
            )
            _features[V_Feature.HeadingError.value] = heading_error           

        if self._use_approximations and obstacle_id not in self._vehicle_arclength_approximations:
            self._vehicle_arclength_approximations[obstacle_id] = {}

        if self._use_approximations:
            # Updating next arclength estimate
            lanelet_tangential_speed = cos(heading_error) * params.state.velocity
            self._vehicle_arclength_approximations[obstacle_id][params.time_step + 1] = lanelet_arclength_abs + params.dt * lanelet_tangential_speed
            if not self._allow_outside_arclengths:
                self._vehicle_arclength_approximations[obstacle_id][params.time_step + 1] = max(
                    0,
                    min(
                        self._vehicle_arclength_approximations[obstacle_id][params.time_step + 1],
                        center_polyline.length
                    )
                )
            self._vehicle_arclength_approximations[obstacle_id][params.time_step] = lanelet_arclength_abs

        return _features

    def __call__(
        self,
        params: V2LFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        features: FeatureDict = {}
        _features=self.compute_features(
            params,
            lanelet=params.lanelet,
            simulation=simulation
        )

        if self._include_longitudinal_abs:
            features[V2L_Feature.V2LLaneletArclengthAbs.value] = _features[V_Feature.LaneletArclengthAbs.value]

        if self._include_longitudinal_rel:
            features[V2L_Feature.V2LLaneletArclengthRel.value] = _features[V_Feature.LaneletArclengthRel.value]

        if self._include_lateral_left:
            features[V2L_Feature.V2LDistLeftBound.value] = _features[V_Feature.DistLeftBound.value]

        if self._include_lateral_right:
            features[V2L_Feature.V2LDistRightBound.value] = _features[V_Feature.DistRightBound.value]

        if self._include_lateral_error:
            features[V2L_Feature.V2LLaneletLateralError.value] = _features[V_Feature.LaneletLateralError.value]

        if self._include_heading_error:
            features[V2L_Feature.V2LHeadingError.value] = _features[V_Feature.HeadingError.value]

        assert all (np.isfinite(f) for f in features.values())

        return features

    def _return_undefined_features(self) -> FeatureDict:
        features: Dict[str, float] = {}

        if self._nan_if_missing:
            if self._include_longitudinal_abs:
                features[V2L_Feature.V2LLaneletArclengthAbs.value] = np.nan
            if self._include_longitudinal_rel:
                features[V2L_Feature.V2LLaneletArclengthRel.value] = np.nan
            if self._include_lateral_left:
                features[V2L_Feature.V2LDistLeftBound.value] = np.nan
            if self._include_lateral_right:
                features[V2L_Feature.V2LDistRightBound.value] = np.nan
            if self._include_lateral_error:
                features[V2L_Feature.V2LLaneletLateralError.value] = np.nan
            if self._include_heading_error:
                features[V2L_Feature.V2LHeadingError.value] = np.nan
        else:
            if self._include_longitudinal_abs:
                features[V_Feature.LaneletArclengthAbs.value] = -1.0
            if self._include_longitudinal_rel:
                features[V_Feature.LaneletArclengthRel.value] = -1.0
            if self._include_lateral_left:
                features[V_Feature.DistLeftBound.value] = -1.0
            if self._include_lateral_right:
                features[V_Feature.DistRightBound.value] = -1.0
            if self._include_lateral_error:
                features[V_Feature.LaneletLateralError.value] = 0.0
            if self._include_heading_error:
                features[V_Feature.HeadingError.value] = 0.0

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        if self._use_approximations:
            self._vehicle_call_count = {}
            self._vehicle_arclength_approximations = {}
