from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, Type, Union

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker
from commonroad_dc.pycrcc import CollisionChecker
from crgeo.common.io_extensions.scenario import backup_scenario  # noqa

from crgeo.simulation.base_simulation import BaseSimulation, BaseSimulationOptions, Unlimited
from crgeo.simulation.ego_simulation.ego_vehicle import EgoVehicle


class UnpopulatedScenarioSimulation(BaseSimulation):
    def __init__(
        self,
        initial_scenario: Union[Scenario, str],
        options: Optional[BaseSimulationOptions] = None
    ) -> None:
        """
        Class for simulations based only on CommonRoad scenarios. Uses TrafficSceneRenderer for rendering.

        Args:
            initial_scenario (Union[Scenario, str]): Initial scenario for this simulation
            options (BaseSimulationOptions): Options for this simulation.
        """
        options = options or BaseSimulationOptions()
        if isinstance(initial_scenario, str):
            initial_scenario, _ = CommonRoadFileReader(filename=initial_scenario).open()
        empty_scenario = deepcopy(initial_scenario)
        for obstacle in empty_scenario.obstacles:
            empty_scenario.remove_obstacle(obstacle)

        # dt from initial_scenario must be used as this was the setting for recording the scenario
        options.dt = initial_scenario.dt
        super().__init__(
            initial_scenario=empty_scenario,
            options=options,
        )

        self._current_collision_checker = create_collision_checker(empty_scenario)

    @property
    def collision_checker(self) -> CollisionChecker:
        return self._current_collision_checker

    def _get_time_step_bounds(self) -> Tuple[int, Type[Unlimited]]:
        return 0, Unlimited

    def _start(self) -> None:
        self.despawn_ego_vehicle(ego_vehicle=None)
        self._time_step_to_obstacles = {}

    def _step(self, ego_vehicle: Optional[EgoVehicle] = None) -> None:
        # Need to integrate ego_vehicle into self._time_step_to_obstacles dictionary
        if ego_vehicle is not None:
            assert ego_vehicle.as_dynamic_obstacle is not None
            self._time_step_to_obstacles[self.current_time_step] = [ego_vehicle.as_dynamic_obstacle]

    def _spawn_ego_vehicle(self, ego_vehicle: EgoVehicle) -> None:
        # Run step first without ego vehicle, otherwise it will be duplicated in ScenarioSimulation._step
        next(self)
        # Then switch to running with ego
        self.lifecycle.run_ego()

    def _close(self) -> None:
        pass

    def _reset(self) -> None:
        self._current_scenario = backup_scenario(self.initial_scenario)
        self._start()
