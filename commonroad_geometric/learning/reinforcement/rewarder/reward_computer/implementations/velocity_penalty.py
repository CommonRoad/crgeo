import math

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class VelocityPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self, 
        reference_velocity: float,
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L1,
        only_upper: bool = False
    ) -> None:
        self._reference_velocity = reference_velocity
        self._weight = weight
        self._loss_type = loss_type
        self._only_upper = only_upper
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        velocity = simulation.ego_vehicle.state.velocity

        if self._only_upper and velocity <= self._reference_velocity:
            return 0.0

        error = velocity - self._reference_velocity
        if self._loss_type == RewardLossMetric.L1:
            loss = abs(error)
        elif self._loss_type == RewardLossMetric.L2:
            loss = error**2
        elif self._loss_type == RewardLossMetric.Gaussian:
            loss = math.exp(-error**2)
        else:
            raise NotImplementedError()
        penalty = -self._weight*loss
        return penalty
