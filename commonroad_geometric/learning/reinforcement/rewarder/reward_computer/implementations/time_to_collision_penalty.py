import math

import numpy as np
import torch

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import MissingFeatureException
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class TimeToCollisionPenaltyRewardComputer(BaseRewardComputer):
    # TODO: Let user choose loss function

    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(self, weight: float, decay_rate: float = 1.0) -> None:
        self._weight = weight
        self._decay_rate = decay_rate
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:

        try:
            # v2v_time_to_closest = data.v2v[V2V_Feature.TimeToClosest.value].squeeze(1)
            # v2v_closest_distance = data.v2v[V2V_Feature.ClosestDistance.value].squeeze(1)
            # v2v_expects_collision = data.v2v[V2V_Feature.ExpectsCollision.value].squeeze(1)
            v2v_time_to_collision = data.v2v[V2V_Feature.TimeToCollision.value].squeeze(1)
        except KeyError:
            raise MissingFeatureException('time_to_collision')

        edge_index = data.v2v.edge_index
        is_ego_mask = data.v.is_ego_mask
        ego_index = torch.where(is_ego_mask)[0].item()
        ego_inc_mask = edge_index[1, :] == ego_index
        ego_inc_index = torch.where(ego_inc_mask)[0]

        # ego_inc_time_to_closest = v2v_time_to_closest[ego_inc_index]
        # ego_inc_closest_distance = v2v_closest_distance[ego_inc_index]
        # ego_inc_expects_collision = v2v_expects_collision[ego_inc_index]
        ego_inc_time_to_collision = v2v_time_to_collision[ego_inc_index]
        ego_inc_time_to_collision.nan_to_num_(nan=np.inf)

        if len(ego_inc_time_to_collision) == 0:
            return 0.0

        min_time_to_collision = ego_inc_time_to_collision.min().item()

        loss = math.exp(-self._decay_rate * min_time_to_collision)

        penalty = -self._weight * loss
        # print(f"min_ttc: {min_time_to_collision:.2f}, loss: {loss:.2f}, penalty: {penalty:.2f}")
        return penalty
