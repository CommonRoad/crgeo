import math

import torch

from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams


def ft_orientation(params: VFeatureParams) -> FeatureDict:
    return {V_Feature.Orientation.value: float(params.state.orientation)}


def ft_veh_state(params: VFeatureParams) -> FeatureDict:
    velocity = torch.tensor([
        params.state.velocity,
        params.state.velocity_y if hasattr(params.state, 'velocity_y') else 0.0,
    ], dtype=torch.float)
    acceleration = torch.tensor([
        params.state.acceleration if "acceleration" in params.state.attributes and params.state.acceleration is not None else 0.0,
        params.state.acceleration_y if "acceleration_y" in params.state.attributes and params.state.acceleration_y is not None else 0.0,
    ], dtype=torch.float)
    return {
        V_Feature.Velocity.value: velocity,
        V_Feature.Acceleration.value: acceleration,
        V_Feature.OrientationVec.value: torch.tensor([
            params.state.orientation,
            math.cos(params.state.orientation),
            math.sin(params.state.orientation),
        ], dtype=torch.float),
        V_Feature.Length.value: params.obstacle.obstacle_shape.length,
        V_Feature.Width.value: params.obstacle.obstacle_shape.width,
    }

def ft_veh_is_clone(params: VFeatureParams) -> FeatureDict:
    velocity = torch.tensor([
        params.state.velocity,
        params.state.velocity_y if hasattr(params.state, 'velocity_y') else 0.0,
    ], dtype=torch.float)
    acceleration = torch.tensor([
        params.state.acceleration if "acceleration" in params.state.attributes and params.state.acceleration is not None else 0.0,
        params.state.acceleration_y if "acceleration_y" in params.state.attributes and params.state.acceleration_y is not None else 0.0,
    ], dtype=torch.float)
    return {
        V_Feature.Velocity.value: velocity,
        V_Feature.Acceleration.value: acceleration,
        V_Feature.OrientationVec.value: torch.tensor([
            params.state.orientation,
            math.cos(params.state.orientation),
            math.sin(params.state.orientation),
        ], dtype=torch.float),
        V_Feature.Length.value: params.obstacle.obstacle_shape.length,
        V_Feature.Width.value: params.obstacle.obstacle_shape.width,
    }