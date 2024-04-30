from __future__ import annotations

import logging
from random import choice
from typing import Optional, Union, cast

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle

from commonroad_geometric.common.io_extensions.planning_problem_set import ObstacleToPlanningProblemException, obstacle_to_planning_problem
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor

logger = logging.getLogger(__name__)


class HijackVehiclePreprocessor(ScenarioPreprocessor):
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
    ) -> None:
        self.random_start_offset = random_start_offset
        self.max_timesteps = max_timesteps
        self.min_timesteps = min_timesteps
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.no_entry_lanelets = no_entry_lanelets
        self.retries = retries
        super().__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        scenario = scenario_bundle.preprocessed_scenario
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
            logger.warning(f"Failed to setup planning problem for scenario id: {scenario_bundle.preprocessed_scenario.scenario_id}, will be excluded during preprocessing")
            return []

        new_planning_problem_set = PlanningProblemSet([planning_problem])

        logger.info(f"Using dynamic obstacle with id {selected_obstacle.obstacle_id} with initial position {planning_problem.initial_state.position} as planning problem for scenario id: "
                    f"{scenario_bundle.preprocessed_scenario.scenario_id}")
        scenario_bundle.preprocessed_planning_problem_set = new_planning_problem_set
        return [scenario_bundle]

