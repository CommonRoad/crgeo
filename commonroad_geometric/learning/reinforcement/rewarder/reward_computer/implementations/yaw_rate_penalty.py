import math
from typing import Optional

import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class YawratePenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self, 
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L2
    ) -> None:
        self._weight = weight
        self._loss_type = loss_type
        self._last_orientation: Optional[float] = None
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        if hasattr(simulation.ego_vehicle.state, 'yaw_rate'):
            yaw_rate = simulation.ego_vehicle.state.yaw_rate
        elif self._last_orientation is not None:
            orientation = simulation.ego_vehicle.state.orientation
            orientation_difference = relative_orientation(self._last_orientation, orientation)
            yaw_rate = orientation_difference / simulation.dt
            self._last_orientation = orientation
        else:
            yaw_rate = 0.0

        error = yaw_rate

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

    def _reset(self) -> None:
        self._last_orientation = None