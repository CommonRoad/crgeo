from typing import Tuple

import pytest
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario


@pytest.fixture(scope="function")
def usa_us101_scenario_planning_problem_set_tuple_xml(usa_us101_path_xml) -> Tuple[Scenario, PlanningProblemSet]:
    # Reading the file once is not sufficient for testing as some test modify the scenario
    scenario, planning_problem_set = CommonRoadFileReader(
        filename=str(usa_us101_path_xml),
    ).open(lanelet_assignment=True)
    return scenario, planning_problem_set


@pytest.fixture(scope="function")
def usa_us101_scenario_xml(usa_us101_scenario_planning_problem_set_tuple_xml) -> Scenario:
    return usa_us101_scenario_planning_problem_set_tuple_xml[0]


@pytest.fixture(scope="function")
def usa_us101_planning_problem_set_xml(usa_us101_scenario_planning_problem_set_tuple_xml) -> PlanningProblemSet:
    return usa_us101_scenario_planning_problem_set_tuple_xml[1]


@pytest.fixture(scope="function")
def usa_us101_planning_problem_xml(usa_us101_planning_problem_set_xml) -> PlanningProblem:
    return usa_us101_planning_problem_set_xml.find_planning_problem_by_id(33)
