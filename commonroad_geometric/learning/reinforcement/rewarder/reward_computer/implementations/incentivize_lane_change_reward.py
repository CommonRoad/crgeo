import numpy as np

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import MissingFeatureException
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class IncentivizeLaneChangeRewardComputer(BaseRewardComputer):

    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(
        self,
        weight: float
    ) -> None:
        self._weight = weight
        self._lanechange_steps: int = 0
        self._desired_lanelet = None
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        reward: float = 0.0
        try:
            desired_lanechange = action[0]
            desired_lanechange_disc = -1 if desired_lanechange < -0.5 else (1 if desired_lanechange > 0.5 else 0)
            desired_lanelet = None
            if len(simulation.current_lanelets) == 0:
                return reward
            current_lanelet = simulation.current_lanelets[0]

            if desired_lanechange_disc == 1 and current_lanelet.adj_left_same_direction:
                desired_lanelet = simulation.current_scenario.lanelet_network.find_lanelet_by_id(
                    current_lanelet.adj_left)

            elif desired_lanechange_disc == -1 and current_lanelet.adj_right_same_direction:
                desired_lanelet = simulation.current_scenario.lanelet_network.find_lanelet_by_id(
                    current_lanelet.adj_right)

            if self._desired_lanelet is None and desired_lanelet is not None:
                self._desired_lanelet = desired_lanelet

            if self._desired_lanelet is not None and (self._desired_lanelet.lanelet_id == current_lanelet.lanelet_id):
                self._lanechange_steps = 1
                return 0.3

            if desired_lanechange_disc == 1 and current_lanelet.adj_left_same_direction or desired_lanechange_disc == - \
                    1 and current_lanelet.adj_right_same_direction:
                self._lanechange_steps = self._lanechange_steps + 1
                if self._lanechange_steps >= 13:
                    self._lanechange_steps = 1
                return self._weight * self._lanechange_steps
            else:
                self._lanechange_steps = 0
            if desired_lanechange_disc == 1 or desired_lanechange_disc == -1:
                return -self._weight
        except KeyError:
            raise MissingFeatureException('_current_lanelets/lanelet_id')
        return reward

    def _reset(self) -> None:
        ...
