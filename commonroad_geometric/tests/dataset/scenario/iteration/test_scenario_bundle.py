from copy import copy
from pathlib import Path
from typing import Tuple

import pytest
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.io_extensions.eq import monkey_patch__eq__
from commonroad_geometric.common.utils.filesystem import save_dill
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle

monkey_patch__eq__()


@pytest.fixture(scope="function")
def scenario_tuple(arg_carcarana_path_xml) -> Tuple[Scenario, PlanningProblemSet]:
    # Reading the file once is not sufficient to test the ScenarioBundle, as we check for object equivalence
    file_reader = CommonRoadFileReader(filename=str(arg_carcarana_path_xml))
    # Test fails without lanelet assignment
    scenario, planning_problem_set = file_reader.open(lanelet_assignment=True)
    return scenario, planning_problem_set


@pytest.fixture()
def scenario(scenario_tuple) -> Scenario:
    scenario, _ = scenario_tuple
    return scenario


@pytest.fixture()
def planning_problem_set(scenario_tuple) -> PlanningProblemSet:
    _, planning_problem_set = scenario_tuple
    return planning_problem_set


def test_scenario_bundle_init(arg_carcarana_path_xml, scenario, planning_problem_set):
    scenario_bundle = ScenarioBundle(
        scenario_path=arg_carcarana_path_xml,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )
    assert scenario_bundle.scenario_path == arg_carcarana_path_xml
    assert scenario_bundle.input_scenario == scenario
    assert scenario_bundle.input_planning_problem_set == planning_problem_set
    assert scenario_bundle.preprocessed_scenario == scenario
    assert scenario_bundle.preprocessed_planning_problem_set == planning_problem_set


def test_scenario_bundle_init_no_planning_problem(arg_carcarana_path_xml, scenario):
    scenario_bundle = ScenarioBundle(
        scenario_path=arg_carcarana_path_xml,
        input_scenario=scenario,
    )
    assert scenario_bundle.scenario_path == arg_carcarana_path_xml
    assert scenario_bundle.input_scenario == scenario
    assert scenario_bundle.preprocessed_scenario == scenario
    assert scenario_bundle.input_planning_problem_set is None
    assert scenario_bundle.preprocessed_planning_problem_set is None


def test_scenario_bundle_copy(arg_carcarana_path_xml, scenario, planning_problem_set):
    scenario_bundle = ScenarioBundle(
        scenario_path=arg_carcarana_path_xml,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )
    copied_scenario_bundle = copy(scenario_bundle)
    # Should be equal and same objects
    assert scenario_bundle.scenario_path == copied_scenario_bundle.scenario_path
    assert scenario_bundle.input_scenario == copied_scenario_bundle.input_scenario
    assert scenario_bundle.input_planning_problem_set == copied_scenario_bundle.input_planning_problem_set
    assert scenario_bundle.scenario_path is copied_scenario_bundle.scenario_path
    assert scenario_bundle.input_scenario is copied_scenario_bundle.input_scenario
    assert scenario_bundle.input_planning_problem_set is copied_scenario_bundle.input_planning_problem_set

    # Should be equal but different objects
    assert scenario_bundle.preprocessed_scenario == copied_scenario_bundle.preprocessed_scenario
    assert scenario_bundle.preprocessed_planning_problem_set == copied_scenario_bundle.preprocessed_planning_problem_set
    assert scenario_bundle.preprocessed_scenario is not copied_scenario_bundle.preprocessed_scenario
    assert (scenario_bundle.preprocessed_planning_problem_set is not
            copied_scenario_bundle.preprocessed_planning_problem_set)


def test_scenario_bundle_hash_deterministic(scenario_bundle):
    hash1 = hash(scenario_bundle)
    hash2 = hash(scenario_bundle)
    assert hash1 == hash2


def test_scenario_bundle_hash_no_change_on_path_change(scenario_bundle):
    old_hash = hash(scenario_bundle)
    scenario_bundle.scenario_path = Path()
    new_hash = hash(scenario_bundle)
    assert old_hash == new_hash


def test_scenario_bundle_hash_changes_on_change(scenario_bundle):
    old_hash = hash(scenario_bundle)
    scenario_bundle.preprocessed_scenario.author = "This test"
    new_hash = hash(scenario_bundle)
    assert old_hash != new_hash


def test_scenario_bundle_eq(scenario_bundle):
    assert scenario_bundle == scenario_bundle


def test_scenario_bundle_not_eq(scenario_bundle):
    scenario_bundle_copy = copy(scenario_bundle)
    scenario_bundle_copy.preprocessed_scenario.author = "This test"
    assert scenario_bundle != scenario_bundle_copy


def test_scenario_bundle_pickling(arg_carcarana_path_xml, scenario, planning_problem_set):
    scenario_bundle = ScenarioBundle(
        scenario_path=arg_carcarana_path_xml,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )
    pickle_path = Path('./output', 'test_scenario_bundle', 'pkl', arg_carcarana_path_xml.stem + '.pkl')
    # Delete the pickle file before testing
    pickle_path.unlink(missing_ok=True)
    save_dill(scenario_bundle, file_path=pickle_path)
    pickled_scenario_bundle = ScenarioBundle.read(pickle_path)

    # Should be equal but different objects
    assert scenario_bundle.scenario_path == pickled_scenario_bundle.scenario_path
    assert scenario_bundle.input_scenario == pickled_scenario_bundle.input_scenario
    assert scenario_bundle.input_planning_problem_set == pickled_scenario_bundle.input_planning_problem_set
    assert scenario_bundle.preprocessed_scenario == pickled_scenario_bundle.preprocessed_scenario
    assert scenario_bundle.preprocessed_planning_problem_set == pickled_scenario_bundle.preprocessed_planning_problem_set

    assert scenario_bundle.scenario_path is not pickled_scenario_bundle.scenario_path
    assert scenario_bundle.input_scenario is not pickled_scenario_bundle.input_scenario
    assert scenario_bundle.input_planning_problem_set is not pickled_scenario_bundle.input_planning_problem_set
    assert scenario_bundle.preprocessed_scenario is not pickled_scenario_bundle.preprocessed_scenario
    assert (scenario_bundle.preprocessed_planning_problem_set is not
            pickled_scenario_bundle.preprocessed_planning_problem_set)


def test_scenario_bundle_read(arg_carcarana_path_xml, scenario, planning_problem_set):
    scenario_bundle = ScenarioBundle.read(arg_carcarana_path_xml)
    assert scenario_bundle.scenario_path == arg_carcarana_path_xml
    assert scenario_bundle.input_scenario == scenario
    assert scenario_bundle.input_planning_problem_set == planning_problem_set
    assert scenario_bundle.preprocessed_scenario == scenario
    assert scenario_bundle.preprocessed_planning_problem_set == planning_problem_set


def test_scenario_bundle_read_lanelet_assignment(
    arg_carcarana_path_xml,
    arg_carcarana_pkl
):
    xml_scenario_bundle = ScenarioBundle.read(
        scenario_path=arg_carcarana_path_xml,
        lanelet_assignment=True
    )
    pkl_scenario_bundle = ScenarioBundle.read(
        scenario_path=arg_carcarana_pkl,
        lanelet_assignment=True
    )
    assert xml_scenario_bundle == pkl_scenario_bundle


def test_scenario_bundle_read_no_lanelet_assignment(
    arg_carcarana_path_xml,
    arg_carcarana_pkl
):
    xml_scenario_bundle = ScenarioBundle.read(
        scenario_path=arg_carcarana_path_xml,
        lanelet_assignment=False
    )
    pkl_scenario_bundle = ScenarioBundle.read(
        scenario_path=arg_carcarana_pkl,
        lanelet_assignment=False
    )
    assert xml_scenario_bundle == pkl_scenario_bundle
