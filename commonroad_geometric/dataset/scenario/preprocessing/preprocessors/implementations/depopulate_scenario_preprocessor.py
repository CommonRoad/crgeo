from __future__ import annotations

from typing import Callable, List, Sequence, Union, cast
import math

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor


class DepopulateScenarioPreprocessor(ScenarioPreprocessor):
    """
    Scenario preprocessor for removing obstacles from scenarios in order to make the traffic less dense.
    """

    def __init__(
        self, 
        depopulator: Union[int, float, Callable[[], int], Callable[[], float]],
        remove_random: bool = True
    ) -> None:
        """
        Initializes preprocessor.

        Args:
            depopulator (Union[int, float, Callable[[int], int], Callable[[int], float]]):
                If int, keep this amount of vehicles
                If float keep this percentage of vehicles
                If callable, first obtain the int or float value as a function of the current call count.
            remove_random (bool, optional): Whether to remove randomly selected vehicles. Defaults to True.
        """
        self._depopulator = depopulator if isinstance(depopulator, (int, float)) else depopulator()
        self._remove_random = remove_random
        super(DepopulateScenarioPreprocessor, self).__init__()

    @property
    def depopulator(self):
        """
        Returns the depopulator value or function.
        """
        return self._depopulator

    @depopulator.setter
    def depopulator(self, depopulator):
        """
        Sets the depopulator value or function.

        Args:
            depopulator (Union[int, float, Callable[[], int], Callable[[], float]]): 
                If int, set this amount of vehicles
                If float, set this percentage of vehicles
                If callable, expect to obtain the int or float value as a function of the current call count.
        """
        print(f"Setting depopulator to {depopulator}")
        self._depopulator = depopulator

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        scenario = scenario_bundle.preprocessed_scenario
        num_vehicles = len(scenario.dynamic_obstacles)

        if num_vehicles == 0:
            return [scenario_bundle]

        depopulate_count: Union[int, float]
        num_obstacles_to_remove: int
        indices_to_remove: Sequence[int]
        obstacles_to_remove: List[DynamicObstacle]

        if isinstance(self._depopulator, (int, float)):
            depopulate_count = self._depopulator
        else:
            depopulate_count = self._depopulator()
        if isinstance(depopulate_count, int):        
            num_obstacles_to_remove = max(0, min(num_vehicles - depopulate_count, num_vehicles))
        else:
            num_obstacles_to_remove = math.ceil(np.clip(1 - depopulate_count, 0.0, 1.0) * num_vehicles)
        if self._remove_random:
            indices_to_remove = cast(Sequence[int], np.random.choice(num_vehicles, num_obstacles_to_remove, replace=False))
        else:
            indices_to_remove = range(num_obstacles_to_remove)
        
        
        obstacles_to_remove = [scenario.dynamic_obstacles[idx] for idx in indices_to_remove]
        for obstacle in obstacles_to_remove:
            try:
                scenario.remove_obstacle(obstacle)
            except Exception as e:
                print(f"Failed to remove obstacle {obstacle.obstacle_id}")

        print(f"Depopulated {len(obstacles_to_remove)}/{num_vehicles}. {depopulate_count=}")

        return [scenario_bundle]
