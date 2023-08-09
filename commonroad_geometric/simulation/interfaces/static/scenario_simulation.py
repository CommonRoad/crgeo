from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object
from commonroad_dc.pycrcc import CollisionChecker  # noqa

from commonroad_geometric.common.io_extensions.scenario import backup_scenario, get_dynamic_obstacles_at_timesteps, get_scenario_timestep_bounds
from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions, SimulationNotYetStartedException
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


@dataclass
class ScenarioSimulationOptions(BaseSimulationOptions):
    """
        remove_ego_vehicle_from_obstacles (bool): Whether to remove the ego vehicle from the current scenario right after init/reset.
    """
    remove_ego_vehicle_from_obstacles: bool = False


class ScenarioSimulation(BaseSimulation):
    def __init__(
        self,
        initial_scenario: Union[Scenario, str],
        options: Optional[ScenarioSimulationOptions] = None,
    ) -> None:
        """
        Class for simulations based only on CommonRoad scenarios. Uses TrafficSceneRenderer for rendering.

        Args:
            initial_scenario (Union[Scenario, str]): Initial scenario for this simulation
            options (ScenarioSimulationOptions): Options for this simulation.
        """
        options = options or ScenarioSimulationOptions()
        options.remove_ego_vehicle_from_obstacles = False
        if isinstance(initial_scenario, str):
            initial_scenario, _ = CommonRoadFileReader(filename=initial_scenario).open()

        # dt from initial_scenario must be used as this was the setting for recording the scenario
        options.dt = initial_scenario.dt
        super().__init__(
            initial_scenario=initial_scenario,
            options=options
        )
        self._options = options  # for correct type annotation
        self._time_variant_collision_checker: CollisionChecker
        self._current_collision_checker_time_step: int

        if options.collision_checking:
            self._time_variant_collision_checker = create_collision_checker(self.current_scenario)
            self._current_collision_checker_time_step = -1
            self._current_collision_checker = None
            self._time_step_to_obstacles = get_dynamic_obstacles_at_timesteps(self.current_scenario)

    @property
    def current_obstacles(self) -> List[DynamicObstacle]:
        return self._time_step_to_obstacles[self.current_time_step]

    @property
    def collision_checker(self) -> CollisionChecker:
        if self._current_time_step is None:
            raise SimulationNotYetStartedException("_current_time_step")
        if self._time_variant_collision_checker is None:
            raise SimulationNotYetStartedException("_time_variant_collision_checker")
        if self._current_collision_checker_time_step != self.current_time_step:
            self._current_collision_checker = self._time_variant_collision_checker.time_slice(self.current_time_step)
            self._current_collision_checker_time_step = self.current_time_step
        return self._current_collision_checker

    def _get_time_step_bounds(self) -> Tuple[int, T_CountParam]:
        scenario = self.initial_scenario
        if scenario.dynamic_obstacles:
            initial_time_step, final_time_step = get_scenario_timestep_bounds(scenario)
            if final_time_step == -1:
                final_time_step = 0
        else:
            initial_time_step, final_time_step = 0, 0
        return initial_time_step, final_time_step

    def _start(self) -> None:
        if self._options.remove_ego_vehicle_from_obstacles:
            # Remove ego vehicle from deep-copied current_scenario, not from initial_scenario
            self.despawn_ego_vehicle(ego_vehicle=None)
        if self._options.collision_checking:
            self._time_variant_collision_checker = create_collision_checker(self.current_scenario)
            self._current_collision_checker_time_step = -1
            self._current_collision_checker = None
        self._time_step_to_obstacles = get_dynamic_obstacles_at_timesteps(self.current_scenario)

    def _spawn_ego_vehicle(self, ego_vehicle: EgoVehicle) -> None:

        # next_ego_state = deepcopy(ego_vehicle.state)
        # next_ego_state.time_step = self.current_time_step + 1
        # ego_vehicle.set_next_state(next_ego_state)
        # self._time_step_to_obstacles[self.current_time_step + 1].append(ego_vehicle.as_dynamic_obstacle)

        # Run step first without ego vehicle, otherwise it will be duplicated in ScenarioSimulation._step
        # next(self)
        # Then switch to running with ego
        self.lifecycle.run_ego()

    def _step(self, ego_vehicle: Optional[EgoVehicle] = None) -> None:
        # Need to integrate ego_vehicle into self._time_step_to_obstacles dictionary
        if ego_vehicle is not None:
            assert ego_vehicle.as_dynamic_obstacle is not None
            self._time_step_to_obstacles[self.current_time_step].append(ego_vehicle.as_dynamic_obstacle)

    def _close(self) -> None:
        self._current_collision_checker_time_step = - 1

    def _reset(self) -> None:
        self._current_collision_checker = None
        if self._options.backup_current_scenario:
            self._current_scenario = backup_scenario(self.initial_scenario) # TODO: Avoid?
        self._start()
