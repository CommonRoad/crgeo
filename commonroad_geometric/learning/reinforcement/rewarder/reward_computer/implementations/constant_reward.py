import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation


class ConstantRewardComputer(BaseRewardComputer):
    """
    Returns a constant reward at each time-step. If negative,
    this corresponds to a living penalty, which can be used to
    incentivise the agent to make progress.
    """

    def __init__(self, reward: float) -> None:
        self.reward = reward
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        return self.reward
