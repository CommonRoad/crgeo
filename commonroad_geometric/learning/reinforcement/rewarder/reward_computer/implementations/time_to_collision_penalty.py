import math
import numpy as np
import torch
from typing import Optional

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.dataset.extraction.traffic.feature_computers import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle.time_to_collision_feature_computer import TimeToCollisionFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, V2VFeatureParams

class TimeToCollisionPenaltyRewardComputer(BaseRewardComputer):
    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(self, weight: float, decay_rate: float = 1.0, distance_threshold: float = 60.0) -> None:
        self._weight = weight
        self._decay_rate = decay_rate
        self._distance_threshold = distance_threshold
        self._feature_computer = None
        super().__init__()

    @property
    def feature_computer(self):
        if self._feature_computer is None:
            self._feature_computer = TimeToCollisionFeatureComputer()
        return self._feature_computer

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: Optional[CommonRoadData],
        observation: T_Observation
    ) -> float:
        if data is not None and hasattr(data, 'v2v') and V2V_Feature.TimeToCollision.value in data.v2v:
            ttc_values = self._extract_ttc_from_data(data)
        else:
            ttc_values = self._compute_ttc_from_simulation(simulation)

        return self._compute_penalty(ttc_values)

    def _extract_ttc_from_data(self, data: CommonRoadData) -> torch.Tensor:
        v2v_time_to_collision = data.v2v[V2V_Feature.TimeToCollision.value].squeeze(1)
        edge_index = data.v2v.edge_index
        is_ego_mask = data.v.is_ego_mask
        ego_index = torch.where(is_ego_mask)[0][-1].item()
        ego_inc_mask = edge_index[1, :] == ego_index
        ego_inc_index = torch.where(ego_inc_mask)[0]
        ego_inc_time_to_collision = v2v_time_to_collision[ego_inc_index]
        return ego_inc_time_to_collision.nan_to_num(nan=np.inf)

    def _compute_ttc_from_simulation(self, simulation: EgoVehicleSimulation) -> np.ndarray:
        ego_vehicle = simulation.ego_vehicle
        ego_state = ego_vehicle.state
        time_to_collision_values = []

        for obstacle in simulation.current_non_ego_obstacles:
            obstacle_state = obstacle.state_at_time(simulation.current_time_step)
            if obstacle_state is not None:
                distance = np.linalg.norm(ego_state.position - obstacle_state.position)
                if distance < self._distance_threshold:
                    params = V2VFeatureParams(
                        distance=distance,
                        source_obstacle=ego_vehicle.as_dynamic_obstacle,
                        source_state=ego_state,
                        source_is_ego_vehicle=True,
                        target_obstacle=obstacle,
                        target_state=obstacle_state,
                        target_is_ego_vehicle=False,
                        ego_state=ego_state,
                        dt=simulation.dt,
                        time_step=simulation.current_time_step
                    )
                    features = self.feature_computer(params, simulation)
                    time_to_collision_values.append(features[V2V_Feature.TimeToCollision.value])

        return np.array(time_to_collision_values)

    def _compute_penalty(self, ttc_values: np.ndarray | torch.Tensor) -> float:
        if len(ttc_values) == 0:
            return 0.0

        min_time_to_collision = np.min(ttc_values)
        loss = math.exp(-self._decay_rate * min_time_to_collision)
        penalty = -self._weight * loss

        return penalty