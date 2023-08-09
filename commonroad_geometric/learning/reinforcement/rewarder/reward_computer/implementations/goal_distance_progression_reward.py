from typing import Optional

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import MissingFeatureException
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class GoalDistanceProgressionRewardComputer(BaseRewardComputer):
    def __init__(
        self, 
        weight: float,
        reward_threshold: Optional[float] = None
    ) -> None:
        self._weight = weight
        self._reward_threshold = reward_threshold or float('inf')
        self._last_goal_distance: Optional[float] = None
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        try:
            goal_distance = data.ego.goal_distance.item()
        except KeyError:
            raise MissingFeatureException('goal_distance')
        if self._last_goal_distance is None:
            progression = 0.0
        else:
            progression = self._last_goal_distance - goal_distance

        self._last_goal_distance = goal_distance
        reward = self._weight*progression
        
        reward = max(0.0, min(reward, self._reward_threshold))

        return reward

    def _reset(self) -> None:
        self._last_goal_distance = None