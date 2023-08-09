import random
import sys
from random import randint
from typing import Optional, Set

from commonroad.scenario.scenario import Scenario

from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.base_traffic_spawner import BaseTrafficSpawner


class ConstantPopulationSpawner(BaseTrafficSpawner):
    """
    Traffic spawner that will enforce an upper limit to the total
    number of traffic participants.
    """

    def __init__(
        self, 
        population_size: int,
        included_lanelet_ids: Optional[Set[int]] = None,
        max_wait_time_steps: int = 30,
        max_spawn_delay: int = 10,
        random_spawn_delay: bool = True,
        p_spawn: Optional[float] = None,
    ):
        super().__init__()
        self._population_size = population_size
        self._waiting = False

        self._max_wait_time_steps = max_wait_time_steps
        self._max_spawn_delay = max_spawn_delay
        self._random_spawn_delay = random_spawn_delay
        self._spawn_delay: float
        self._set_spawn_delay()
        self._last_num_vehicles: int = -1
        self._last_spawn_timestep: int = -1
        self._lanelet_id_to_spawn: Optional[int] = None
        self._current_time_step: Optional[int] = None
        self._p_spawn = p_spawn if p_spawn is not None else True
        super(ConstantPopulationSpawner, self).__init__(included_lanelet_ids=included_lanelet_ids)

    def _set_spawn_delay(self) -> None:
        if self._random_spawn_delay:
            self._spawn_delay = randint(1, self._max_spawn_delay)
        else:
            self._spawn_delay = self._max_wait_time_steps

    def _spawn_density(self, scenario: Scenario, time_step: int, lanelet_id: int) -> float:
        if time_step != self._current_time_step:
            self._lanelet_id_to_spawn = random.choice(self._entry_lanelet_ids)
            self._current_time_step = time_step

        if lanelet_id != self._lanelet_id_to_spawn:
            return 0.0

        num_vehicles = sum(1 for obstacle in scenario.dynamic_obstacles if obstacle.obstacle_id != -1)
        if num_vehicles != self._last_num_vehicles:
            self._waiting = False

        time_steps_since_last_spawn = time_step - self._last_spawn_timestep if self._last_spawn_timestep >= 0 else sys.maxsize

        if time_steps_since_last_spawn >= self._spawn_delay and \
                (not self._waiting or time_steps_since_last_spawn > self._max_wait_time_steps):
            if num_vehicles < self._population_size:
                self._waiting = True
                self._last_num_vehicles = num_vehicles
                self._last_spawn_timestep = time_step
                self._set_spawn_delay()
                return self._p_spawn
        return 0.0

    def reset(self, scenario: Scenario) -> None:
        super().reset(scenario=scenario)
        self._last_num_vehicles = -1
        self._last_spawn_timestep = -1
        self._lanelet_id_to_spawn = None
        self._current_time_step = None
