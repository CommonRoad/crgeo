from __future__ import annotations
from typing import Optional, Tuple

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from crgeo.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet


class VehicleFilterPreprocessor(BaseScenarioPreprocessor):

    def __init__(
        self, 
    ) -> None:
        super(VehicleFilterPreprocessor, self).__init__()

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:

        for obstacle in scenario.dynamic_obstacles:
            if obstacle.obstacle_type not in {ObstacleType.CAR, ObstacleType.TRUCK, ObstacleType.BUS}:
                scenario.remove_obstacle(obstacle)
        
        return scenario, planning_problem_set
