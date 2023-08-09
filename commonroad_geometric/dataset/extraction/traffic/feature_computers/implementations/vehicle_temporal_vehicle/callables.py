import numpy as np
import torch
from typing import List

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.common.torch_utils.geometry import rotate_2d, translate_rotate_2d
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VTVFeatureParams


def ft_rel_state_vtv(params: VTVFeatureParams) -> List[float]:
    source_orientation = params.data.v.orientation[params.past_obstacle_idx].item()
    target_orientation = params.data.v.orientation[params.curr_obstacle_idx].item()
    source_pos = params.data.v.pos[params.past_obstacle_idx]
    target_pos = params.data.v.pos[params.curr_obstacle_idx]
    source_velocity = params.data.v.velocity[params.past_obstacle_idx]
    target_velocity = params.data.v.velocity[params.curr_obstacle_idx]
    source_acceleration = params.data.v.acceleration[params.past_obstacle_idx]
    target_acceleration = params.data.v.acceleration[params.curr_obstacle_idx]

    rel_orientation = relative_orientation(source_orientation, target_orientation)

    target_pos_in_source_coord_sys = translate_rotate_2d(
        x=target_pos,
        t=-source_pos, 
        r=-rel_orientation,
    )
    rel_position = target_pos_in_source_coord_sys
    distance = np.linalg.norm(rel_position)

    target_vel_in_source_coord_sys = rotate_2d(
        target_velocity, 
        r=-rel_orientation
    )
    rel_velocity = torch.tensor([
        target_vel_in_source_coord_sys[0] - source_velocity[0],
        target_vel_in_source_coord_sys[1] - source_velocity[1],
    ], dtype=torch.float)

    target_acc_in_source_coord_sys = rotate_2d(
        target_acceleration,
        r=-rel_orientation
    )
    rel_acceleration = torch.tensor([
        target_acc_in_source_coord_sys[0] - source_acceleration[0],
        target_acc_in_source_coord_sys[1] - source_acceleration[1],
    ], dtype=torch.float)

    return {
        "rel_orientation": rel_orientation,
        "rel_position": rel_position,
        "distance": distance,
        "rel_velocity": rel_velocity,
        "rel_acceleration": rel_acceleration
    }