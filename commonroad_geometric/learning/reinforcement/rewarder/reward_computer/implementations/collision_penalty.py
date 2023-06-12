from typing import Optional

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class CollisionPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        penalty: float,
        not_at_fault_penalty: Optional[float] = None,
        speed_multiplier: bool = False,
        max_speed: float = 15.0,
        speed_bias: float = 1.0,
    ) -> None:
        self.penalty = penalty
        self.not_at_fault_penalty = not_at_fault_penalty if not_at_fault_penalty is not None else penalty
        self.speed_multiplier = speed_multiplier
        self.max_speed = max_speed
        self.speed_bias = speed_bias
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        collision_struct = simulation.check_if_has_collision()
        if collision_struct.collision:
            penalty = self.penalty if collision_struct.ego_at_fault else self.not_at_fault_penalty
            if self.speed_multiplier:
                speed = abs(simulation.ego_vehicle.state.velocity)
                speed_multiplier = np.clip(speed + self.speed_bias, 0.0, self.max_speed) / self.max_speed
                penalty *= speed_multiplier
            return penalty
        return 0.0
