from typing import Optional

import numpy as np

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import MissingFeatureException
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class LaneChangeRewardComputer(BaseRewardComputer):

    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(
        self,
        dense: bool,
        weight: float
    ) -> None:
        self._dense = dense
        self._weight = weight
        self._last_lane_changes_required: Optional[int] = None
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        try:
            lane_changes_required: int = data.ego.lane_changes_required[-1].item()
        except KeyError:
            raise MissingFeatureException('lane_changes_required')
        loss = 0
        if self._dense:
            loss = lane_changes_required
        else:
            if self._last_lane_changes_required is not None:
                if lane_changes_required > self._last_lane_changes_required:
                    loss = 1
                elif lane_changes_required < self._last_lane_changes_required:
                    loss = -1
            self._last_lane_changes_required = lane_changes_required
        reward = -self._weight * loss
        return reward

    def _reset(self) -> None:
        self._last_lane_changes_required = None
