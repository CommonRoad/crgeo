from typing import Optional, Set, TYPE_CHECKING

import numpy as np
from commonroad.scenario.scenario import Scenario, ScenarioID

from crgeo.simulation.interfaces.interactive.traffic_spawning.base_traffic_spawner import BaseTrafficSpawner


class ConstantRateSpawner(BaseTrafficSpawner):

    def __init__(
        self, 
        p_spawn: float = 0.05,
        included_lanelet_ids: Optional[Set[int]] = None,
        max_vehicles: Optional[int] = None,
        normalize_network_size: bool = False
    ) -> None:
        super(ConstantRateSpawner, self).__init__(included_lanelet_ids=included_lanelet_ids)
        self._p_spawn = p_spawn
        self._max_vehicles = max_vehicles
        self._normalize_network_size = normalize_network_size
        self._cached_scenario_id: Optional[ScenarioID] = None
        self._cached_network_size: Optional[int] = None

    def _spawn_density_vectorized(self, scenario: Scenario, time_step: int) -> Optional[np.ndarray]:
        if self._normalize_network_size:
            raise NotImplementedError()
            # if self._cached_scenario_id is None or scenario.scenario_id != self._cached_scenario_id:
            #     self._cached_network_size = len([l for l in scenario.lanelet_network.lanelets if not l.predecessor])
            #     self._cached_scenario_id = scenario.scenario_id
            # norm_factor = 1 / self._cached_network_size

        if self._max_vehicles is not None:
            num_vehicles = sum(1 for obstacle in scenario.dynamic_obstacles if obstacle.obstacle_id != -1)
            if num_vehicles >= self._max_vehicles:
                return np.zeros(len(self._entry_lanelet_ids))
        return np.full(len(self._entry_lanelet_ids), self._p_spawn)
