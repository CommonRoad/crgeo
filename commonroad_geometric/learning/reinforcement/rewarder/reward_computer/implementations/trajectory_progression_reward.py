from typing import Optional

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

ARCLENGTH_SCALE = 1 / 100


class TrajectoryProgressionRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
        reward_threshold: Optional[float] = None,
        delta_threshold: Optional[float] = None,
        dynamic_weight: float = 1.0,
        relative_arclength: bool = True
    ) -> None:
        self._weight = weight
        self._reward_threshold = reward_threshold or float('inf')
        self._delta_threshold = delta_threshold
        self._dynamic_weight = dynamic_weight
        self._relative_arclength = relative_arclength
        self._last_arclength: Optional[float] = None
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData
    ) -> float:
        assert simulation.ego_vehicle.ego_route is not None
    
        reward = 0.0
        if simulation.ego_vehicle.ego_route.planning_problem_path_polyline is not None:
            ego_position = simulation.ego_vehicle.state.position
            ego_trajectory_polyline = simulation.ego_vehicle.ego_route.planning_problem_path_polyline
            if self._relative_arclength:
                arclength = ego_trajectory_polyline.get_projected_arclength(ego_position, relative=True)
            else:
                arclength = ego_trajectory_polyline.get_projected_arclength(ego_position, relative=False) * ARCLENGTH_SCALE
            delta_arclength = 0.0 if self._last_arclength is None else (arclength - self._last_arclength) / simulation.dt
            if self._delta_threshold is not None:
                delta_arclength = min(self._delta_threshold, delta_arclength)
            self._last_arclength = arclength
            reward = self._weight * (self._dynamic_weight * delta_arclength + (1 - self._dynamic_weight) * arclength)
        return max(0.0, min(reward, self._reward_threshold))

    def _reset(self) -> None:
        self._last_arclength = None
