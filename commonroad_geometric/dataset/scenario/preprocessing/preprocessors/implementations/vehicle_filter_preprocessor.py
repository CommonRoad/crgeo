from __future__ import annotations
from typing import Optional, Tuple

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet


class VehicleFilterPreprocessor(ScenarioPreprocessor):

    def __init__(
        self, 
    ) -> None:
        super(VehicleFilterPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        for obstacle in scenario_bundle.preprocessed_scenario.dynamic_obstacles:
            if obstacle.obstacle_type not in {ObstacleType.CAR, ObstacleType.TRUCK, ObstacleType.BUS}:
                scenario_bundle.preprocessed_scenario.remove_obstacle(obstacle)
        
        return [scenario_bundle]
