import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class PathFollowingRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        max_speed: float,
        cross_track_error_sensitivity: float
    ) -> None:
        self.max_speed = max_speed
        self.cross_track_error_sensitivity = cross_track_error_sensitivity
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        heading_error = observation["path_observation"][1]
        cross_track_error = observation["path_observation"][-2]
        rel_velocity = min(1.0, simulation.ego_vehicle.state.velocity / self.max_speed)
        cross_track_performance = np.exp(-self.cross_track_error_sensitivity * np.abs(cross_track_error))

        path_reward = (1 + np.cos(heading_error) * rel_velocity) * (1 + cross_track_performance) - 1

        # print(f"{heading_error=:.2f}, {cross_track_error=:.2f}, {rel_velocity=:.2f}, {cross_track_performance=:.2f} => {path_reward=:.2f}")

        return path_reward
