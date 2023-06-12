import itertools
from typing import Tuple, Iterable, Optional

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class TrafficFilterer(BaseScenarioFilterer):

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

    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario

        if len(scenario.dynamic_obstacles) < self.min_vehicles:
            return False

        if self.min_avg_mean_speed is not None or self.min_avg_min_speed is not None:
            avg_velocities = np.zeros((len(scenario.dynamic_obstacles)))
            min_velocities = np.zeros((len(scenario.dynamic_obstacles)))
            for i, obstacle in enumerate(scenario.dynamic_obstacles):
                state_list = obstacle.prediction.trajectory.state_list
                velocities = np.array([s.velocity for s in state_list])
                avg_velocities[i] = velocities.mean()
                min_velocities[i] = velocities.min()
            if self.min_avg_mean_speed is not None and avg_velocities.mean() <= self.min_avg_mean_speed:
                return False
            if self.min_avg_min_speed is not None and min_velocities.mean() <= self.min_avg_mean_speed:
                return False
        
        return True
