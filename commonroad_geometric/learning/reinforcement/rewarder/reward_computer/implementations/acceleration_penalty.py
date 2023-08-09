import math

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class AccelerationPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self, 
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L1,
        cutoff = 10.0
    ) -> None:
        self._weight = weight
        self._loss_type = loss_type
        self._cutoff = cutoff
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        acceleration = abs(simulation.ego_vehicle.state.acceleration)
        acceleration = min(self._cutoff, acceleration) * simulation.dt

        if self._loss_type == RewardLossMetric.L1:
            loss = abs(acceleration)
        elif self._loss_type == RewardLossMetric.L2:
            loss = acceleration**2
        elif self._loss_type == RewardLossMetric.Gaussian:
            loss = math.exp(-acceleration**2)
        else:
            raise NotImplementedError()
        penalty = -self._weight*loss
        return penalty
