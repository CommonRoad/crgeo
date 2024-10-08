import math

import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class HeadingErrorPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
        loss_type: RewardLossMetric = RewardLossMetric.L1,
        wrong_direction_penalty: float = 0.0
    ) -> None:
        self.weight = weight
        self.loss_type = loss_type
        self.wrong_direction_penalty = wrong_direction_penalty
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        current_state = simulation.ego_vehicle.state
        path_direction = simulation.ego_route.extended_path_polyline.get_projected_direction(current_state.position)
        heading_error = relative_orientation(
            current_state.orientation,
            path_direction
        )

        if not math.isfinite(heading_error):
            heading_error = 0.0

        if self.loss_type == RewardLossMetric.L1:
            loss = abs(heading_error)
        elif self.loss_type == RewardLossMetric.L2:
            loss = heading_error**2
        elif self.loss_type == RewardLossMetric.Gaussian:
            loss = math.exp(-heading_error**2)
        else:
            raise NotImplementedError()
        penalty = -self.weight * loss

        if abs(heading_error) > np.pi / 2:
            penalty += self.wrong_direction_penalty

        return penalty
