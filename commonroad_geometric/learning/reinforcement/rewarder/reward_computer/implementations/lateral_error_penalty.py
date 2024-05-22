import math

import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import MissingFeatureException, RewardLossMetric
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class LateralErrorPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L1
    ) -> None:
        self._weight = weight
        self._loss_type = loss_type
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        try:
            lateral_error = data.ego.lanelet_lateral_error.item()
        except AttributeError:
            current_state = simulation.ego_vehicle.state
            try:
                lanelet_id = simulation.simulation.obstacle_id_to_lanelet_id[simulation.ego_vehicle.obstacle_id][0]
            except IndexError:
                lateral_error = 0.0
            else:
                path = simulation.simulation.get_lanelet_center_polyline(lanelet_id)
                lateral_error = path.get_lateral_distance(
                    current_state.position,
                    linear_projection=simulation.simulation.options.linear_lanelet_projection
                )

        if not math.isfinite(lateral_error):
            lateral_error = 0.0

        if self._loss_type == RewardLossMetric.L1:
            loss = abs(lateral_error)
        elif self._loss_type == RewardLossMetric.L2:
            loss = lateral_error**2
        elif self._loss_type == RewardLossMetric.Gaussian:
            loss = math.exp(-lateral_error**2)
        else:
            raise NotImplementedError()

        penalty = -self._weight * loss
        return penalty
