from typing import List, Optional, Set

import numpy as np
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.random.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.base_traffic_spawner import BaseTrafficSpawner


class OrnsteinUhlenbeckSpawner(BaseTrafficSpawner):

    def __init__(
        self,
        # TODO: set appropriate defaults
        p_mean: float = 0.5,
        p_sigma: float = 0.25,
        p_theta: float = 0.2,
        p_dt: float = 1e-1,
        included_lanelet_ids: Optional[Set[int]] = None
    ) -> None:
        super(OrnsteinUhlenbeckSpawner, self).__init__(included_lanelet_ids=included_lanelet_ids)
        self._p_mean = p_mean
        self._p_sigma = p_sigma
        self._p_theta = p_theta
        self._p_dt = p_dt
        self._spawn_processes: Optional[List[OrnsteinUhlenbeckProcess]] = None

    def reset(self, scenario: Scenario) -> None:
        super().reset(scenario=scenario)
        self._spawn_processes = [
            OrnsteinUhlenbeckProcess(
                mean=self._p_mean,
                sigma=self._p_sigma,
                theta=self._p_theta,
                dt=self._p_dt
            ) for _ in self._entry_lanelet_ids
        ]

    def _spawn_density_vectorized(self, scenario: Scenario, time_step: int) -> Optional[np.ndarray]:
        assert self._spawn_processes is not None
        return np.array([process().item() for process in self._spawn_processes])
