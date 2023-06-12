from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import gym.spaces
import gym.spaces
import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from projects.geometric_models.lane_occupancy.models.occupancy.occupancy_model import DEFAULT_PATH_LENGTH

logger = logging.getLogger(__name__)


@dataclass
class FeatureScaleOptions:
    fmin: Optional[float]
    fmax: Optional[float]


class Features(IntEnum):
    GOAL_DISTANCE = 0
    VELOCITY = 1
    STEERING_ANGLE = 2
    HEADING_ERROR = 3
    LATERAL_ERROR = 4
    LOOK_AHEAD_CURVATURES = 5
    LOOK_AHEAD_YAW_DIFFS = 6
    Z_EGO = 7
    ACCELERATION = 8
    HEADING_ERROR_ABS = 9
    LATERAL_ERROR_ABS = 10
    LOOK_AHEAD_CURVATURES_ABS = 11
    LOOK_AHEAD_YAW_DIFFS_ABS = 12
    STEERING_ANGLE_ABS = 13


class EncodingObserver(BaseObserver):
    N_FIXED_EGO_FEATURES = 6
    N_FIXED_EGO_FEATURES_LONG = 3
    N_FIXED_SIGNED_EGO_FEATURES = 3
    LOOK_AHEAD_TIMES = np.array([0.1, 0.3, 0.7, 1.2, 2.0, 3.0, 5.0])

    FEATURE_SCALING_OPTIONS = {
        Features.GOAL_DISTANCE: FeatureScaleOptions(0.0, DEFAULT_PATH_LENGTH),
        Features.VELOCITY: FeatureScaleOptions(0.0, 22.0),
        Features.STEERING_ANGLE_ABS: FeatureScaleOptions(0.0, 0.4),
        Features.STEERING_ANGLE: FeatureScaleOptions(-0.4, 0.4),
        Features.HEADING_ERROR_ABS: FeatureScaleOptions(0.0, np.pi/4),
        Features.LATERAL_ERROR_ABS: FeatureScaleOptions(0.0, 2.5),
        Features.LOOK_AHEAD_CURVATURES_ABS: FeatureScaleOptions(0.0, 0.2),
        Features.LOOK_AHEAD_YAW_DIFFS_ABS: FeatureScaleOptions(0.0, np.pi/2),
        Features.HEADING_ERROR: FeatureScaleOptions(-np.pi/4, np.pi/4),
        Features.LATERAL_ERROR: FeatureScaleOptions(-2.5, 2.5),
        Features.LOOK_AHEAD_CURVATURES: FeatureScaleOptions(-0.2, 0.2),
        Features.LOOK_AHEAD_YAW_DIFFS: FeatureScaleOptions(-np.pi/2, np.pi/2),
        Features.Z_EGO: FeatureScaleOptions(None, None),
        Features.ACCELERATION: FeatureScaleOptions(0.0, 20.0)
    }

    def __init__(
        self,
        only_abs_geom_features: bool = False,
        look_ahead_times: Optional[np.ndarray] = None,
        only_longitudinal_features: bool = False
    ) -> None:
        assert not (only_abs_geom_features and only_longitudinal_features)
        self.only_abs_geom_features = only_abs_geom_features
        self.only_longitudinal_features = only_longitudinal_features
        self.look_ahead_times = look_ahead_times if look_ahead_times is not None else EncodingObserver.LOOK_AHEAD_TIMES
        self._n_encoding_features: int
        self._n_features: int
        super().__init__()

    def setup(self, dummy_data: CommonRoadData) -> gym.spaces.Space:
        self._n_encoding_features = dummy_data.z_ego_route.shape[0]
        if self.only_abs_geom_features:
            n_features = sum((
                self._n_encoding_features,
                EncodingObserver.N_FIXED_EGO_FEATURES,
                2*len(self.look_ahead_times)
            ))
        elif self.only_longitudinal_features:
            n_features = sum((
                self._n_encoding_features,
                EncodingObserver.N_FIXED_EGO_FEATURES_LONG,
            ))
        else:
            n_features = sum((
                self._n_encoding_features,
                EncodingObserver.N_FIXED_EGO_FEATURES,
                EncodingObserver.N_FIXED_SIGNED_EGO_FEATURES,
                4*len(self.look_ahead_times)
            ))
        self._n_features = n_features
        observation_space = gym.spaces.Box(-np.inf, np.inf, (n_features, ), dtype=np.float32)
        return observation_space

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> Dict[str, np.ndarray]:
        opt = EncodingObserver.FEATURE_SCALING_OPTIONS
        F = Features

        current_state = ego_vehicle_simulation.ego_vehicle.state
        path = ego_vehicle_simulation.ego_route.extended_path_polyline
        assert path is not None
        arclength = path.get_projected_arclength(current_state.position)
        path_direction = path.get_direction(arclength)

        velocity = current_state.velocity
        goal_distance = data.ego.goal_distance_long.item()
        acceleration = current_state.acceleration
        velocity = EncodingObserver._scale_feature(opt[F.VELOCITY].fmin, opt[F.VELOCITY].fmax, velocity)
        acceleration = EncodingObserver._scale_feature(opt[F.ACCELERATION].fmin, opt[F.ACCELERATION].fmax, acceleration)
        goal_distance = EncodingObserver._scale_feature(opt[F.GOAL_DISTANCE].fmin, opt[F.GOAL_DISTANCE].fmax, goal_distance)

        try:
            z_ego = data.z_ego_route.detach().numpy()
        except AttributeError:
            logger.warning("Encoding property 'z_ego_route' missing on data. Setting zeros")
            z_ego = np.zeros((self._n_encoding_features, ), dtype=np.float32)

        if self.only_longitudinal_features:
            x_scalars = np.array([
                goal_distance,
                velocity,
                acceleration
            ])

            x = np.concatenate([
                x_scalars,
                z_ego
            ], axis=-1)
            return x

        look_ahead_locations = current_state.velocity * self.look_ahead_times
        look_ahead_curvatures = np.empty((len(self.look_ahead_times), ), dtype=np.float32)
        look_ahead_yaw_diffs = np.empty((len(self.look_ahead_times), ), dtype=np.float32)
        for i, sample_location in enumerate(look_ahead_locations):
            look_ahead_arclength = arclength + sample_location
            curvature = path.get_curvature(look_ahead_arclength)
            look_ahead_yaw_diff = relative_orientation(path_direction, path.get_direction(look_ahead_arclength))
            look_ahead_curvatures[i] = curvature
            look_ahead_yaw_diffs[i] = look_ahead_yaw_diff
        look_ahead_yaw_diffs_abs = np.abs(look_ahead_yaw_diffs)
        look_ahead_curvatures_abs = np.abs(look_ahead_curvatures)
        heading_error = relative_orientation(current_state.orientation, path.get_direction(arclength))
        heading_error_abs = abs(heading_error)
        lateral_error = path.get_lateral_distance(current_state.position)
        lateral_error_abs = abs(lateral_error)
        steering_angle = current_state.steering_angle
        steering_angle_abs = abs(steering_angle)

        steering_angle = EncodingObserver._scale_feature(opt[F.STEERING_ANGLE].fmin, opt[F.STEERING_ANGLE].fmax, steering_angle)
        steering_angle_abs = EncodingObserver._scale_feature(opt[F.STEERING_ANGLE_ABS].fmin, opt[F.STEERING_ANGLE_ABS].fmax, steering_angle_abs)
        heading_error = EncodingObserver._scale_feature(opt[F.HEADING_ERROR].fmin, opt[F.HEADING_ERROR].fmax, heading_error)
        heading_error_abs = EncodingObserver._scale_feature(opt[F.HEADING_ERROR_ABS].fmin, opt[F.HEADING_ERROR_ABS].fmax, heading_error_abs)
        lateral_error = EncodingObserver._scale_feature(opt[F.LATERAL_ERROR].fmin, opt[F.LATERAL_ERROR].fmax, lateral_error)
        lateral_error_abs = EncodingObserver._scale_feature(opt[F.LATERAL_ERROR_ABS].fmin, opt[F.LATERAL_ERROR_ABS].fmax, lateral_error_abs)
        look_ahead_curvatures = EncodingObserver._scale_feature(opt[F.LOOK_AHEAD_CURVATURES].fmin, opt[F.LOOK_AHEAD_CURVATURES].fmax, look_ahead_curvatures)
        look_ahead_yaw_diffs = EncodingObserver._scale_feature(opt[F.LOOK_AHEAD_YAW_DIFFS].fmin, opt[F.LOOK_AHEAD_YAW_DIFFS].fmax, look_ahead_yaw_diffs)
        look_ahead_curvatures_abs = EncodingObserver._scale_feature(opt[F.LOOK_AHEAD_CURVATURES_ABS].fmin, opt[F.LOOK_AHEAD_CURVATURES_ABS].fmax, look_ahead_curvatures_abs)
        look_ahead_yaw_diffs_abs = EncodingObserver._scale_feature(opt[F.LOOK_AHEAD_YAW_DIFFS_ABS].fmin, opt[F.LOOK_AHEAD_YAW_DIFFS_ABS].fmax, look_ahead_yaw_diffs_abs)

        x_scalars = np.array([
            goal_distance,
            velocity,
            steering_angle_abs,
            heading_error_abs,
            lateral_error_abs,
            acceleration
        ])
        x_scalars_signed = np.array([
            steering_angle,
            heading_error,
            lateral_error,
        ])
        if self.only_abs_geom_features:
            x = np.concatenate([
                x_scalars,
                look_ahead_curvatures_abs,
                look_ahead_yaw_diffs_abs,
                z_ego
            ], axis=-1)
        else:
            x = np.concatenate([
                x_scalars,
                x_scalars_signed,
                look_ahead_curvatures,
                look_ahead_yaw_diffs,
                look_ahead_curvatures_abs,
                look_ahead_yaw_diffs_abs,
                z_ego
            ], axis=-1)

        return x

    @staticmethod
    def _scale_feature(
        fmin: Optional[float],
        fmax: Optional[float],
        value: float
    ) -> float:
        if fmin is None or fmax is None:
            return value
        clipped_value = np.clip(value, fmin, fmax)
        mean = 0.5 * (fmax + fmin) # TODO
        interval = 0.5 * (fmax - fmin)
        scaled_value = (clipped_value - mean) / interval
        return scaled_value
