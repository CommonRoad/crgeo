import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class FrictionViolationPenaltyRewardComputer(BaseRewardComputer):
    def __init__(self, penalty: float) -> None:
        self._penalty = penalty
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        if simulation.check_if_violates_friction():
            return self._penalty
        return 0.0
