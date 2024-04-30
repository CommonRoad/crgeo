from typing import Optional

import numpy as np

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class TrafficFilter(ScenarioFilter):
    """
    Rejects scenarios with insufficient traffic.
    """

    def __init__(
        self,
        min_avg_mean_speed: Optional[float] = None,
        min_avg_min_speed: Optional[float] = None,
        min_vehicles: int = 13
    ):
        self.min_avg_mean_speed = min_avg_mean_speed
        self.min_avg_min_speed = min_avg_min_speed
        self.min_vehicles = min_vehicles
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        dynamic_obstacles = scenario_bundle.preprocessed_scenario.dynamic_obstacles

        if len(dynamic_obstacles) < self.min_vehicles:
            return False

        if self.min_avg_mean_speed is not None or self.min_avg_min_speed is not None:
            avg_velocities = np.zeros((len(dynamic_obstacles)))
            min_velocities = np.zeros((len(dynamic_obstacles)))
            for i, obstacle in enumerate(dynamic_obstacles):
                state_list = obstacle.prediction.trajectory.state_list
                velocities = np.array([s.velocity for s in state_list])
                avg_velocities[i] = velocities.mean()
                min_velocities[i] = velocities.min()
            if self.min_avg_mean_speed is not None and avg_velocities.mean() <= self.min_avg_mean_speed:
                return False
            if self.min_avg_min_speed is not None and min_velocities.mean() <= self.min_avg_mean_speed:
                return False

        return True
