from typing import List, Optional, Set, TYPE_CHECKING

import numpy as np

from commonroad.scenario.scenario import Scenario
from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.simulation.interfaces.interactive.traffic_spawning.base_traffic_spawner import BaseTrafficSpawner


class OrnsteinUhlenbeckProcess(AutoReprMixin):
    """
    A Ornstein Uhlenbeck noise process designed to approximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    """

    def __init__(self, mean: float, sigma: float, theta: float = 0.15, dt: float = 1e-2):
        super().__init__()
        self._theta = np.array([theta])
        self._mu = np.array([mean])
        self._sigma = sigma
        self._dt = dt
        self._noise_prev: np.ndarray
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = self._noise_prev + self._theta * (self._mu - self._noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self._noise_prev = noise
        return noise

    def reset(self) -> None:
        self._noise_prev = np.zeros_like(self._mu)


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
