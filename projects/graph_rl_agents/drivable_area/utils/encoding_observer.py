from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Optional, Literal

import gymnasium
import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.common.math import scale_feature
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.ego.goal_alignment_feature_computer import GoalAlignmentComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VFeatureParams
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from projects.graph_rl_agents.drivable_area.utils.surrounding_observation import SurroundingObservation

logger = logging.getLogger(__name__)


@dataclass
class FeatureScaleOptions:
    fmin: Optional[float]
    fmax: Optional[float]


@unique
class Features(IntEnum):
    GOAL_DISTANCE = 0
    VELOCITY = 1
    ACCELERATION = 2
    STEERING_ANGLE = 3
    STEERING_ANGLE_ABS = 4
    HEADING_ERROR = 5
    HEADING_ERROR_ABS = 6
    LATERAL_ERROR = 7
    LATERAL_ERROR_ABS = 8


class EncodingObserver(BaseObserver):
    FEATURE_SCALING_OPTIONS = {
        Features.GOAL_DISTANCE: FeatureScaleOptions(0.0, 100.0),
        Features.VELOCITY: FeatureScaleOptions(0.0, 40.0),
        Features.ACCELERATION: FeatureScaleOptions(0.0, 20.0),
        Features.STEERING_ANGLE: FeatureScaleOptions(-0.4, 0.4),
        Features.STEERING_ANGLE_ABS: FeatureScaleOptions(0.0, 0.4),
        Features.HEADING_ERROR: FeatureScaleOptions(-np.pi / 4, np.pi / 4),
        Features.HEADING_ERROR_ABS: FeatureScaleOptions(0.0, np.pi / 4),
        Features.LATERAL_ERROR: FeatureScaleOptions(-2.5, 2.5),
        Features.LATERAL_ERROR_ABS: FeatureScaleOptions(0.0, 2.5),
    }

    N_FIXED_EGO_FEATURES = len(FEATURE_SCALING_OPTIONS) + 4 # (4 goal alignment features)

    def __init__(
        self,
        observation_type: Literal["encoding", "lidar_circle", "lane_circle"]  = "encoding"
    ) -> None:
        self._n_encoding_features: int = 0

        self._goal_alignment_computer = GoalAlignmentComputer(
            include_lane_changes_required=True,
            logarithmic=False,
            closeness_transform=True
        )

        assert observation_type in {"encoding", "lidar_circle", "lane_circle"}
        self.observation_type = observation_type

        if observation_type != "encoding":
            self._surrounding_observer = SurroundingObservation({
                'observe_lane_circ_surrounding': observation_type == "lane_circle",
                'observe_lidar_circle_surrounding': observation_type == "lidar_circle"
            })

        super().__init__()

    def setup(self, dummy_data: CommonRoadData) -> gymnasium.Space:
        if self.observation_type == "encoding":
            self._n_encoding_features = dummy_data.encoding.shape[-1]
        else:
            self._n_encoding_features = 2*6 # lidar or lane circ hardcoded
        
        observation_space = gymnasium.spaces.Box(
            -np.inf,
            np.inf,
            (EncodingObserver.N_FIXED_EGO_FEATURES + self._n_encoding_features,),
            dtype=np.float32
        )
        return observation_space

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        opt = EncodingObserver.FEATURE_SCALING_OPTIONS
        F = Features

        if self.observation_type != "encoding":
            if not ego_vehicle_simulation.current_lanelets:
                z_ego = np.zeros(self._n_encoding_features)
            else:
                if not hasattr(ego_vehicle_simulation.simulation, '_connected_lanelet_dict'):
                    ego_vehicle_simulation.simulation._connected_lanelet_dict = ego_vehicle_simulation.simulation.get_all_connected_lanelets()
                surrounding_obs, _ = self._surrounding_observer.observe(
                    scenario=ego_vehicle_simulation.current_scenario,
                    ego_vehicle=ego_vehicle_simulation.ego_vehicle,
                    time_step=ego_vehicle_simulation.current_time_step,
                    connected_lanelet_dict=ego_vehicle_simulation.simulation._connected_lanelet_dict,
                    ego_lanelet=ego_vehicle_simulation.current_lanelets[0],
                    local_ccosy=ego_vehicle_simulation.ego_route.navigator.ccosy_list[0]
                )
                if self.observation_type == "lane_circle":
                    z_ego = np.concatenate([surrounding_obs['lane_based_v_rel']/10, surrounding_obs['lane_based_p_rel']/10])
                else:
                    raise NotImplementedError()
        else:
            try:
                z_ego = data.encoding.squeeze(0).detach().numpy()
            except AttributeError:
                logger.warning("Encoding property 'encoding' missing on data. Setting zeros")
                z_ego = np.zeros((self._n_encoding_features,), dtype=np.float32)

        current_state = ego_vehicle_simulation.ego_vehicle.state
        velocity = current_state.velocity
        steering_angle = current_state.steering_angle
        steering_angle_abs = abs(steering_angle)
        velocity = current_state.velocity
        acceleration = current_state.acceleration if current_state.acceleration is not None else 0.0

        try:
            lanelet_id = ego_vehicle_simulation.simulation.obstacle_id_to_lanelet_id[
                ego_vehicle_simulation.ego_vehicle.obstacle_id][0]
            lanelet_path = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(lanelet_id)
            arclength = lanelet_path.get_projected_arclength(current_state.position)
            path_direction = lanelet_path.get_direction(arclength)
            heading_error = relative_orientation(
                current_state.orientation,
                path_direction
            )
            heading_error_abs = abs(heading_error)
            lateral_error = lanelet_path.get_lateral_distance(
                current_state.position
            )
            lateral_error_abs = abs(lateral_error)
        except IndexError:
            heading_error = 0.0
            heading_error_abs = 0.0
            lateral_error = 0.0
            lateral_error_abs = 0.0

        goal_alignment_dict = self._goal_alignment_computer(
            params=VFeatureParams(
                dt=ego_vehicle_simulation.dt,
                time_step=ego_vehicle_simulation.current_time_step,
                obstacle=ego_vehicle_simulation.ego_vehicle.as_dynamic_obstacle,
                state=current_state,
                is_ego_vehicle=True,
                ego_state=current_state,
                ego_route=ego_vehicle_simulation.ego_route
            ),
            simulation=ego_vehicle_simulation.simulation
        )
        goal_distance = goal_alignment_dict[V_Feature.GoalDistanceLongitudinal.value]

        # goal_distance = scale_feature(opt[F.GOAL_DISTANCE].fmin, opt[F.GOAL_DISTANCE].fmax, goal_distance)
        velocity = scale_feature(opt[F.VELOCITY].fmin, opt[F.VELOCITY].fmax, velocity)
        acceleration = scale_feature(opt[F.ACCELERATION].fmin, opt[F.ACCELERATION].fmax, acceleration)
        steering_angle = scale_feature(opt[F.STEERING_ANGLE].fmin, opt[F.STEERING_ANGLE].fmax, steering_angle)
        steering_angle_abs = scale_feature(opt[F.STEERING_ANGLE_ABS].fmin,
                                           opt[F.STEERING_ANGLE_ABS].fmax, steering_angle_abs)
        heading_error = scale_feature(opt[F.HEADING_ERROR].fmin, opt[F.HEADING_ERROR].fmax, heading_error)
        heading_error_abs = scale_feature(opt[F.HEADING_ERROR_ABS].fmin,
                                          opt[F.HEADING_ERROR_ABS].fmax, heading_error_abs)
        lateral_error = scale_feature(opt[F.LATERAL_ERROR].fmin, opt[F.LATERAL_ERROR].fmax, lateral_error)
        lateral_error_abs = scale_feature(opt[F.LATERAL_ERROR_ABS].fmin,
                                          opt[F.LATERAL_ERROR_ABS].fmax, lateral_error_abs)

        x_scalars = np.array([
            goal_distance,
            velocity,
            acceleration,
            steering_angle,
            steering_angle_abs,
            heading_error,
            heading_error_abs,
            lateral_error,
            lateral_error_abs,
            goal_alignment_dict['goal_distance_lat'],
            goal_alignment_dict['goal_heading_error'],
            goal_alignment_dict['lane_changes_required'],
            goal_alignment_dict['lane_change_dir_required'],
        ])

        x = np.concatenate([
            x_scalars,
            z_ego
        ], axis=-1)

        # print(f"{z_ego=}, {x_scalars=}")

        return x
