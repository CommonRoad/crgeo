from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from copy import deepcopy

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import State, CustomState, InitialState
from commonroad_geometric.common.geometry.helpers import make_valid_orientation

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet


class CloneVehicleTrajectoriesPreprocessor(ScenarioPreprocessor):

    def __init__(
        self, 
        position_noise: float = 0.0,
        orientation_noise: float | str = 0.0
    ) -> None:
        self.position_noise = position_noise
        self.orientation_noise = orientation_noise
        super(CloneVehicleTrajectoriesPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        for obstacle in scenario_bundle.preprocessed_scenario.dynamic_obstacles:
            cloned_trajectory = []

            for state in obstacle.prediction.trajectory.state_list:
                position = state.position + np.random.normal(loc=np.array([0, 0]), scale=self.position_noise*np.ones((2,)))
                if self.orientation_noise == "uniform":
                    orientation_noise = np.random.uniform(-np.pi, np.pi)
                else:
                    orientation_noise = np.random.normal(loc=0.0, scale=self.orientation_noise)
                orientation = make_valid_orientation(state.orientation + orientation_noise)
                copied_state = deepcopy(state)
                copied_state.position = position
                copied_state.orientation = orientation
                cloned_trajectory.append(copied_state)

            clone = DynamicObstacle(
                obstacle_id=scenario_bundle.preprocessed_scenario.generate_object_id(),
                obstacle_type=obstacle.obstacle_type,
                obstacle_shape=obstacle.obstacle_shape,
                initial_state=obstacle.initial_state,
                prediction=TrajectoryPrediction(
                    trajectory=Trajectory(
                        initial_time_step=obstacle.prediction.initial_time_step,
                        state_list=cloned_trajectory
                    ),
                    shape=obstacle.obstacle_shape
                )
            )
            clone.__dict__['_is_clone'] = True
            scenario_bundle.preprocessed_scenario.add_objects(clone)
        
        return [scenario_bundle]
