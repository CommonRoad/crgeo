from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.io_extensions.hash import hash_planning_problem_set, hash_scenario
from commonroad_geometric.common.utils.filesystem import FileFormatNotSupportedError, load_dill


def already_assigned(dynamic_obstacles: List[DynamicObstacle]):
    for dynamic_obstacle in dynamic_obstacles:
        prediction = dynamic_obstacle.prediction
        if prediction.center_lanelet_assignment is None:
            return False
        if prediction.shape_lanelet_assignment is None:
            return False
        if not dynamic_obstacle.prediction.center_lanelet_assignment:
            return False
        if not dynamic_obstacle.prediction.shape_lanelet_assignment:
            return False


def un_assign_obstacles(dynamic_obstacles: List[DynamicObstacle]):
    for dynamic_obstacle in dynamic_obstacles:
        dynamic_obstacle.initial_center_lanelet_ids = None
        dynamic_obstacle.initial_shape_lanelet_ids = None
        dynamic_obstacle.prediction.center_lanelet_assignment = None
        dynamic_obstacle.prediction.shape_lanelet_assignment = None


@dataclass(init=False, eq=False)
class ScenarioBundle:
    """
    Container object yielded by ScenarioIterator.
    """
    scenario_path: Path
    input_scenario: Scenario
    input_planning_problem_set: Optional[PlanningProblemSet]
    preprocessed_scenario: Scenario
    preprocessed_planning_problem_set: Optional[PlanningProblemSet]

    def __init__(
        self,
        scenario_path: Path,
        input_scenario: Scenario,
        input_planning_problem_set: Optional[PlanningProblemSet] = None,
        preprocessed_scenario: Optional[Scenario] = None,
        preprocessed_planning_problem_set: Optional[PlanningProblemSet] = None,
    ) -> None:
        self.scenario_path = scenario_path
        self.input_scenario = input_scenario
        self.input_planning_problem_set = input_planning_problem_set
        # Invariant: Must either be initialized with scenario or an already preprocessed scenario
        if preprocessed_scenario is not None:
            self.preprocessed_scenario = preprocessed_scenario
        else:
            self.preprocessed_scenario = deepcopy(self.input_scenario)

        # Invariant: Must be initialized when input_planning_problem_set or preprocessed_planning_problem_set is not None
        if preprocessed_planning_problem_set is not None:
            self.preprocessed_planning_problem_set = preprocessed_planning_problem_set
        elif input_planning_problem_set is not None:
            self.preprocessed_planning_problem_set = deepcopy(self.input_planning_problem_set)
        else:
            self.preprocessed_planning_problem_set = None

    def __copy__(self):
        """
        Returns:
            shallow copy of ScenarioBundle where only the preprocessed_scenario and
            preprocessed_planning_problem_set are deep-copied
        """
        copied_bundle = type(self).__new__(self.__class__)
        deepcopied_keys = {"preprocessed_scenario", "preprocessed_planning_problem_set"}
        for key, value in self.__dict__.items():
            if key in deepcopied_keys:
                copied_bundle.__dict__[key] = deepcopy(value)
            else:
                copied_bundle.__dict__[key] = value
        return copied_bundle

    def __eq__(self, other: ScenarioBundle):
        """
        Compares this ScenarioBundle ignoring the scenario path as not
        to confuse two otherwise equal bundles.

        Returns:
            True if the scenario bundles are equal, False otherwise.
        """
        if not isinstance(other, ScenarioBundle):
            return False

        return self.input_scenario == other.input_scenario and \
            self.input_planning_problem_set == other.input_planning_problem_set and \
            self.preprocessed_scenario == other.preprocessed_scenario and \
            self.preprocessed_planning_problem_set == other.preprocessed_planning_problem_set

    def __hash__(self):
        """
        Creates a hash of this ScenarioBundle ignoring the scenario path as not
        to confuse two otherwise equal bundles.

        Returns:
            hash of ScenarioBundle excluding the scenario path
        """
        return hash(
            (
                hash_scenario(scenario=self.input_scenario),
                hash_planning_problem_set(self.input_planning_problem_set),
                hash_scenario(scenario=self.preprocessed_scenario),
                hash_planning_problem_set(self.preprocessed_planning_problem_set)
            )
        )

    @staticmethod
    def read(
        scenario_path: Path,
        lanelet_assignment: bool = True
    ) -> ScenarioBundle:
        """
        Args:
            scenario_path (Path): path to scenario file or pickled ScenarioBundle
            lanelet_assignment (bool): Whether to assign dynamic obstacles to lanelets in the case of .xml files.
                                       Defaults to True.

        Returns:
            ScenarioBundle created or loaded from scenario path

        Raises:
            FileNotFoundError: If file does not exist.
            FileFormatNotSupportedError: If file format is not supported. Supports: {'.xml', '.pkl'}.
        """
        filename = str(scenario_path)
        match scenario_path.suffix:
            case ".pkl":
                if scenario_path.is_file():
                    scenario_bundle = cast(ScenarioBundle, load_dill(file_path=scenario_path))
                    if lanelet_assignment:
                        if not already_assigned(scenario_bundle.input_scenario.dynamic_obstacles):
                            scenario_bundle.input_scenario.assign_obstacles_to_lanelets()
                            scenario_bundle.preprocessed_scenario.assign_obstacles_to_lanelets()
                    else:
                        un_assign_obstacles(scenario_bundle.input_scenario.dynamic_obstacles)
                        un_assign_obstacles(scenario_bundle.preprocessed_scenario.dynamic_obstacles)
                    return scenario_bundle
                else:
                    raise FileNotFoundError(f"File {scenario_path.absolute()} does not exist! "
                                            f"Make sure that the specified path leads to a pickled scenario bundle.")
            case ".xml":
                file_reader = CommonRoadFileReader(filename=filename)
            case _format:
                raise FileFormatNotSupportedError(f"File format '{_format}' is not supported! "
                                                  f"ScenarioBundle.read supports: {{'.xml', '.pkl'}}")

        scenario, planning_problem_set = file_reader.open(lanelet_assignment=lanelet_assignment)
        scenario_bundle = ScenarioBundle(
            scenario_path=scenario_path,
            input_scenario=scenario,
            input_planning_problem_set=planning_problem_set,
        )
        return scenario_bundle
