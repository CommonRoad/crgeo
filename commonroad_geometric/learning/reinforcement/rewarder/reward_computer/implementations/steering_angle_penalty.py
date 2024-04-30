import math

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class SteeringAnglePenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L1
    ) -> None:
        self._weight = weight
        self._loss_type = loss_type
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        steering_angle = action[0]

        if self._loss_type == RewardLossMetric.L1:
            loss = abs(steering_angle)
        elif self._loss_type == RewardLossMetric.L2:
            loss = steering_angle**2
        elif self._loss_type == RewardLossMetric.Gaussian:
            loss = math.exp(-steering_angle**2)
        else:
            raise NotImplementedError()
        penalty = -self._weight * loss
        return penalty
