import numpy as np
import torch

from commonroad_geometric.common.geometry.helpers import relative_orientation, rotate_2d, translate_rotate_2d
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, V2VFeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature
from commonroad_geometric.simulation.base_simulation import BaseSimulation


def ft_same_lanelet(params: V2VFeatureParams, simulation: BaseSimulation) -> FeatureDict:
    source_lanelet = simulation.get_obstacle_lanelet(params.source_obstacle)
    target_lanelet = simulation.get_obstacle_lanelet(params.target_obstacle)
    same_lanelet = source_lanelet.lanelet_id == target_lanelet.lanelet_id if source_lanelet is not None and target_lanelet is not None else False
    return {V2V_Feature.SameLanelet.value: int(same_lanelet)}


def ft_rel_state(params: V2VFeatureParams, simulation: BaseSimulation) -> FeatureDict:
    rel_position = params.source_state.position - params.target_state.position
    distance = np.linalg.norm(rel_position)
    rel_velocity = params.source_state.velocity - params.target_state.velocity
    rel_orientation = relative_orientation(params.target_state.orientation, params.source_state.orientation)
    return {
        V2V_Feature.RelativePosition.value: torch.from_numpy(rel_position),
        V2V_Feature.Distance.value: distance,
        V2V_Feature.RelativeVelocity.value: rel_velocity,
        V2V_Feature.RelativeOrientation.value: rel_orientation
    }


def ft_rel_state_ego(params: V2VFeatureParams, simulation: BaseSimulation) -> FeatureDict:
    # all values are computed in vehicle-fixed coordinate system of the source vehicle

    rel_orientation = relative_orientation(params.source_state.orientation, params.target_state.orientation)

    target_pos_in_source_coord_sys = translate_rotate_2d(
        x=params.target_state.position,
        t=-params.source_state.position, r=-rel_orientation,
    )
    rel_position = target_pos_in_source_coord_sys
    distance = np.linalg.norm(rel_position)

    source_velocity_y = params.source_state.velocity_y if "velocity_y" in params.source_state.attributes else 0.0
    target_velocity_y = params.target_state.velocity_y if "velocity_y" in params.target_state.attributes else 0.0
    source_acceleration = params.source_state.acceleration if "acceleration" in params.source_state.attributes else 0.0
    target_acceleration = params.target_state.acceleration if "acceleration" in params.target_state.attributes else 0.0
    source_acceleration_y = params.source_state.acceleration_y if "acceleration_y" in params.source_state.attributes else 0.0
    target_acceleration_y = params.target_state.acceleration_y if "acceleration_y" in params.target_state.attributes else 0.0

    target_vel_in_target_coord_sys = np.array([
        params.target_state.velocity,
        target_velocity_y,
    ])
    target_vel_in_source_coord_sys = rotate_2d(target_vel_in_target_coord_sys, r=-rel_orientation)
    rel_velocity = torch.tensor([
        target_vel_in_source_coord_sys[0] - params.source_state.velocity,
        target_vel_in_source_coord_sys[1] - source_velocity_y,
    ], dtype=torch.float)

    target_acc_in_target_coord_sys = np.array([
        target_acceleration,
        target_acceleration_y,
    ])
    target_acc_in_source_coord_sys = rotate_2d(target_acc_in_target_coord_sys, r=-rel_orientation)
    rel_acceleration = torch.tensor([
        target_acc_in_source_coord_sys[0] - source_acceleration,
        target_acc_in_source_coord_sys[1] - source_acceleration_y,
    ], dtype=torch.float)

    return {
        V2V_Feature.RelativeOrientationEgo.value: rel_orientation,
        V2V_Feature.RelativePositionEgo.value: torch.from_numpy(rel_position),
        V2V_Feature.DistanceEgo.value: distance,
        V2V_Feature.RelativeVelocityEgo.value: rel_velocity,
        V2V_Feature.RelativeAccelerationEgo.value: rel_acceleration,
    }
