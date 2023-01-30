from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union, cast

from typing import TYPE_CHECKING

import numpy as np
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from crgeo.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor


class DepopulateScenarioPreprocessor(BaseScenarioPreprocessor):
    """
    Scenario preprocessor for removing obstacles from scenarios in order to make the traffic less dense.
    """

    def __init__(
        self, 
        depopulator: Union[int, float, Callable[[int], int], Callable[[int], float]],
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
        self._depopulator = depopulator
        self._remove_random = remove_random
        super(DepopulateScenarioPreprocessor, self).__init__()

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        num_vehicles = len(scenario.dynamic_obstacles)

        if num_vehicles == 0:
            return scenario

        depopulate_count: Union[int, float]
        num_obstacles_to_remove: int
        indices_to_remove: Sequence[int]
        obstacles_to_remove: List[DynamicObstacle]

        if isinstance(self._depopulator, (int, float)):
            depopulate_count = self._depopulator
        else:
            depopulate_count = self._depopulator(self.call_count)
        if isinstance(depopulate_count, int):        
            num_obstacles_to_remove = max(0, min(num_vehicles - depopulate_count, num_vehicles))
        else:
            num_obstacles_to_remove = int(np.clip(1 - depopulate_count, 0.0, 1.0) * num_vehicles)
        if self._remove_random:
            indices_to_remove = cast(Sequence[int], np.random.choice(num_vehicles, num_obstacles_to_remove, replace=False))
        else:
            indices_to_remove = range(num_obstacles_to_remove)
        obstacles_to_remove = [scenario.dynamic_obstacles[idx] for idx in indices_to_remove]
        for obstacle in obstacles_to_remove:
            scenario.remove_obstacle(obstacle)

        return scenario, planning_problem_set
