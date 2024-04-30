import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class StillStandingPenaltyRewardComputer(BaseRewardComputer):
    def __init__(self, penalty: float, velocity_threshold: float = 2.0) -> None:
        self._penalty = penalty
        self._velocity_threshold = velocity_threshold
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        if abs(simulation.ego_vehicle.state.velocity) < self._velocity_threshold:
            return self._penalty
        return 0.0
