from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union, List, cast
from random import choice

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_geometric.common.io_extensions.planning_problem import ObstacleToPlanningProblemException, obstacle_to_planning_problem
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor


class HijackVehiclePreprocessor(BaseScenarioPreprocessor):
    """
    Selects a random vehicle present in the scenario and turns it into a planning problem
    (while removing it from the scene).
    """

    def __init__(
        self,
        random_start_offset: bool = True,
        max_timesteps: Optional[Union[int, float]] = 50,
        min_timesteps: Optional[int] = 50,
        min_distance: Optional[float] = 15.0,
        max_distance: Optional[float] = 20.0,
        no_entry_lanelets: bool = True,
        retries: int = 200,
        raise_on_failure: bool = True
    ) -> None:
        self.random_start_offset = random_start_offset
        self.max_timesteps = max_timesteps
        self.min_timesteps = min_timesteps
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.no_entry_lanelets = no_entry_lanelets
        self.retries = retries
        self.raise_on_failure = raise_on_failure
        super().__init__()

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        for _ in range(self.retries):
            selected_obstacle = cast(DynamicObstacle, choice(scenario.dynamic_obstacles))
            try:
                planning_problem = obstacle_to_planning_problem(
                    obstacle=selected_obstacle,
                    lanelet_network=scenario.lanelet_network,
                    planning_problem_id=0,
                    random_start_offset=self.random_start_offset,
                    max_timesteps=self.max_timesteps,
                    min_timesteps=self.min_timesteps,
                    min_distance=self.min_distance,
                    max_distance=self.max_distance,
                    no_entry_lanelets=self.no_entry_lanelets
                )
            except ObstacleToPlanningProblemException:
                continue
            scenario.remove_obstacle(selected_obstacle)
            break
        else:
            if self.raise_on_failure:
                raise ValueError("Failed to setup planning problem")
            return scenario, planning_problem_set

        new_planning_problem_set = PlanningProblemSet([planning_problem])

        print(planning_problem.initial_state.position, planning_problem.initial_state.position, selected_obstacle.obstacle_id)

        return scenario, new_planning_problem_set
