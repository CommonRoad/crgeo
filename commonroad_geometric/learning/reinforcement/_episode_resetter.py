from __future__ import annotations

import logging
import timeit
from threading import Thread
from typing import Generic, Optional, TYPE_CHECKING

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.threading.delayed_executor import DelayedExecutor
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.simulation.base_simulation import T_SimulationOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

if TYPE_CHECKING:
    from commonroad_geometric.learning.reinforcement.commonroad_gym_env import CommonRoadGymEnv

logger = logging.getLogger(__name__)

ASYNC_DELAY_MULTIPLIER_INCREASE = 1.03  # TODO: Provide these as arguments
ASYNC_DELAY_MULTIPLIER_DECREASE = 0.9


class _EpisodeResetter(Generic[T_SimulationOptions], AutoReprMixin):
    """
    Asynchronous gym environment episode resetter for spreading compute time of .reset() calls across preceding step() calls,
    alleviating delays when training agent in parallel environments.
    """

    def __init__(
        self,
        env: CommonRoadGymEnv[T_SimulationOptions],
        async_resets: bool = True,
        injected_async_delay: float = 0.2,
        auto_update_delay: bool = True
    ) -> None:
        self._env = env
        self._active_setup_task: Optional[Thread] = None
        self._next_ego_vehicle_simulation: Optional[EgoVehicleSimulation] = None
        self._injected_async_delay = injected_async_delay
        self._async_resets = async_resets
        self._auto_update_delay = auto_update_delay
        self._delayed_executor = DelayedExecutor(
            disabled=not async_resets,
            delay=self._injected_async_delay
        )
        logger.debug(
            f"Initialized _EpisodeResetter with async_resets={async_resets}, injected_async_delay={injected_async_delay}")

    def on_init(self) -> None:
        """
        Setting up first scenario synchronously since no background task has yet been run.
        Should be called from the gym environment's __init__ method.
        """
        self._setup_next()
        self._set_next()

    def on_step(self) -> None:
        """
        Starts loading the second scenario (special case)
        """
        if self._async_resets and not self._env.has_started:
            self._setup_next_async()

    def on_reset(self) -> None:
        """
        Should be called from the gym environment's reset method.
        """
        self._end_current()
        if self._async_resets:
            ready_on_time = self.ready
            if not ready_on_time:
                self._delayed_executor.disable()
            if self._auto_update_delay:
                if ready_on_time:
                    self._delayed_executor.multiply_sleep_duration(ASYNC_DELAY_MULTIPLIER_INCREASE)
                else:
                    self._delayed_executor.multiply_sleep_duration(ASYNC_DELAY_MULTIPLIER_DECREASE)
            self._join_setup_thread()
            if not ready_on_time:
                self._delayed_executor.enable()
        else:
            self._setup_next()
        self._set_next()
        if self._async_resets:
            self._setup_next_async()

    @property
    def ready(self) -> bool:
        if not self._async_resets or self._active_setup_task is None:
            return False
        return not self._active_setup_task.is_alive()

    def _end_current(self) -> None:
        """
        Closes current ego vehicle simulation instance if it exists.
        """
        if self._env._ego_vehicle_simulation is not None:
            self._env._ego_vehicle_simulation.close()

    def _setup_next(self) -> None:
        """
        Loads next scenario bundle and initializes ego vehicle simulation instance.
        """
        start_time = timeit.default_timer()

        success = False
        while not success:
            next_scenario_bundle: ScenarioBundle = self._delayed_executor(
                self._env.scenario_iterable.__next__
            )
            logger.debug(f"Completed subtask 1/3: Retrieved next scenario bundle")
            next_ego_vehicle_simulation: EgoVehicleSimulation = self._delayed_executor(
                self._env.ego_vehicle_simulation_factory.create,
                scenario_bundle=next_scenario_bundle
            )
            logger.debug(f"Completed subtask 2/3: Initialized next ego simulation")
            success = self._delayed_executor(
                next_ego_vehicle_simulation.start
            )
            if not success:
                logger.debug(f"Failed subtask 3/3: Started simulation (retrying)")
        logger.debug(f"Completed subtask 3/3: Started simulation")
        self._next_ego_vehicle_simulation = next_ego_vehicle_simulation
        elapsed_time = timeit.default_timer() - start_time
        logger.info(f"Completed {'async' if self._async_resets else 'sync'} setup of next scenario {self._next_ego_vehicle_simulation.current_scenario.scenario_id} after {elapsed_time:.2f} seconds (took {self._env.step_counter} steps)")

    def _setup_next_async(self) -> None:
        """
        Spawn new worker thread preparing for next episode reset.
        """
        logger.debug("Spawned async worker preparing next episode reset")
        self._active_setup_task = Thread(target=self._setup_next)
        self._active_setup_task.start()

    def _join_setup_thread(self) -> None:
        """
        Waits for active backgrund worker to finish.
        """
        assert self._active_setup_task is not None
        self._active_setup_task.join()

    def _set_next(self) -> None:
        assert self._next_ego_vehicle_simulation is not None
        self._env._ego_vehicle_simulation = self._next_ego_vehicle_simulation
        logger.debug(f"Set next environment simulation")
