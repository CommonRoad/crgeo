from __future__ import annotations

import logging
from abc import ABC
from typing import List, Optional, Set, Union

import numpy as np
import numpy.random
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin

logger = logging.getLogger(__name__)


class TrafficSpawnerNotResetException(Exception):
    pass


class BaseTrafficSpawner(ABC, AutoReprMixin, StringResolverMixin):

    def __init__(
        self,
        included_lanelet_ids: Optional[Set[int]] = None
    ) -> None:
        self._included_lanelet_ids = included_lanelet_ids
        self._entry_lanelet_ids: List[int] = []
        self._was_reset = False

    def reset(self, scenario: Scenario) -> None:
        self._entry_lanelet_ids = []
        for lanelet in scenario.lanelet_network.lanelets:
            if not lanelet.predecessor and \
                    (self._included_lanelet_ids is None or lanelet.lanelet_id in self._included_lanelet_ids):
                self._entry_lanelet_ids.append(lanelet.lanelet_id)
        self._was_reset = True

    def spawn(self, dt: float, time_step: int, scenario: Scenario) -> Set[int]:
        """
        Returns vehicles to be spawned in the current scenario

        Args:
            scenario (Scenario): Current CommonRoad scenario.
        Returns:
            Set[int]: Set of lanelet ids for which a vehicle should be spawned.
        """

        if not self._was_reset:
            raise TrafficSpawnerNotResetException(f"{self.__class__.__name__} instance has not been reset yet")

        spawn_set: Set[int]
        p_spawn_density: Union[float, np.ndarray, bool, None]

        p_spawn_density = self._spawn_density_vectorized(scenario=scenario, time_step=time_step)

        if p_spawn_density is not None:
            assert p_spawn_density.shape == (len(self._entry_lanelet_ids),)
            if p_spawn_density.dtype == bool:
                spawn_mask = p_spawn_density
            else:  # dtype == float
                spawn_mask = numpy.random.random_sample(len(self._entry_lanelet_ids)) <= p_spawn_density

            spawn_lanelet_ids = numpy.array(self._entry_lanelet_ids)[spawn_mask]
            spawn_lanelet_ids = spawn_lanelet_ids.tolist()  # convert to list of Python int data type
            spawn_set = set(spawn_lanelet_ids)

        else:
            spawn_set = set()
            for lanelet_id in self._entry_lanelet_ids:
                p_spawn_density = self._spawn_density(scenario=scenario, time_step=time_step, lanelet_id=lanelet_id)
                if isinstance(p_spawn_density, bool):
                    if p_spawn_density:
                        spawn_set.add(lanelet_id)
                else:
                    p_spawn = p_spawn_density * dt
                    if p_spawn >= 1.0 or (p_spawn > 0.0 and numpy.random.random_sample() <= p_spawn):
                        spawn_set.add(lanelet_id)

        # if spawn_set:
        #     logger.debug(f"Spawning vehicles on lanelets: {spawn_set}")

        return spawn_set

    def _spawn_density_vectorized(self, scenario: Scenario, time_step: int) -> Optional[np.ndarray]:
        """
        Return spawn probability for all lanelet ids in _entry_lanelet_ids as a numpy array or None
        if this method is not implemented and the _spawn_density method should be used instead.
        """
        return None

    def _spawn_density(self, scenario: Scenario, time_step: int, lanelet_id: int) -> Union[float, bool]:
        """
        Return spawn probability for the specified lanelet_id 

        Args:
            lanelet_id (int): Lanelet id.

        Returns:
            float: Spawn probability.
        """
        raise NotImplementedError()
