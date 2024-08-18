from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet


class ComputeVehicleVelocitiesPreprocessor(ScenarioPreprocessor):

    def __init__(
        self, 
    ) -> None:
        super(ComputeVehicleVelocitiesPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        dt = scenario_bundle.preprocessed_scenario.dt
        for obstacle in scenario_bundle.preprocessed_scenario.dynamic_obstacles:

            for t in range(len(obstacle.prediction.trajectory.state_list) - 1):
                state = obstacle.prediction.trajectory.state_list[t]
                next_state = obstacle.prediction.trajectory.state_list[t + 1]
                displacement = np.linalg.norm(next_state.position - state.position)
                velocity = displacement / dt
                state.velocity = velocity
            try:
                next_state.velocity = velocity
            except UnboundLocalError:
                pass

        return [scenario_bundle]
