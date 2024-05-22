import shutil
from pathlib import Path

import pytest

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.scenario_bundle_writer import ScenarioBundleWriter


@pytest.fixture
def output_directory():
    return Path('./output', Path(__file__).stem)


def test_scenario_bundle_writer(
    scenario_bundle: ScenarioBundle,
    output_directory: Path,
):
    # Delete output directory before every test to prevent side effects
    shutil.rmtree(output_directory, ignore_errors=True)

    path = scenario_bundle.scenario_path
    expected_output_path = output_directory / f"{path.stem}_{hash(scenario_bundle)}.pkl"
    # Delete file at expected output path before test
    expected_output_path.unlink(missing_ok=True)
    assert not expected_output_path.is_file()

    scenario_bundle_writer = ScenarioBundleWriter(
        output_directory=output_directory
    )
    result = scenario_bundle_writer(scenario_bundle)

    assert result
    assert len(result) == 1
    # Returns the unmodified scenario bundle
    assert result[0] is scenario_bundle
    assert expected_output_path.is_file()

    saved_scenario_bundle = ScenarioBundle.read(scenario_path=expected_output_path)
    # Path should have been modified
    assert saved_scenario_bundle.scenario_path == expected_output_path
    assert saved_scenario_bundle.scenario_path != scenario_bundle.scenario_path

    # Everything else should not have been modified during conversion
    assert saved_scenario_bundle.input_scenario == scenario_bundle.input_scenario
    assert saved_scenario_bundle.input_planning_problem_set == scenario_bundle.input_planning_problem_set
    assert saved_scenario_bundle.preprocessed_scenario == scenario_bundle.preprocessed_scenario
    assert saved_scenario_bundle.preprocessed_planning_problem_set == scenario_bundle.preprocessed_planning_problem_set


def test_scenario_bundle_writer_no_output_directory(
    scenario_bundle: ScenarioBundle,
    output_directory: Path,
):
    # Delete output directory before every test to prevent side effects
    shutil.rmtree(output_directory, ignore_errors=True)

    path = scenario_bundle.scenario_path
    expected_output_path = output_directory / f"{path.stem}_{hash(scenario_bundle)}.pkl"
    # Delete file at expected output path before test
    expected_output_path.unlink(missing_ok=True)
    assert not expected_output_path.is_file()

    scenario_bundle_writer = ScenarioBundleWriter()

    # Modify scenario_path to pretend that this scenario bundle comes from the
    # expected output directory, could be replaced by path mock
    scenario_bundle.scenario_path = output_directory / path.name
    result = scenario_bundle_writer(scenario_bundle)

    assert result
    assert len(result) == 1
    # Returns the unmodified scenario bundle
    assert result[0] is scenario_bundle
    assert expected_output_path.is_file()

    saved_scenario_bundle = ScenarioBundle.read(scenario_path=expected_output_path)
    # Path should have been modified
    assert saved_scenario_bundle.scenario_path == expected_output_path
    assert saved_scenario_bundle.scenario_path != scenario_bundle.scenario_path

    # Everything else should not have been modified during conversion
    assert saved_scenario_bundle.input_scenario == scenario_bundle.input_scenario
    assert saved_scenario_bundle.input_planning_problem_set == scenario_bundle.input_planning_problem_set
    assert saved_scenario_bundle.preprocessed_scenario == scenario_bundle.preprocessed_scenario
    assert saved_scenario_bundle.preprocessed_planning_problem_set == scenario_bundle.preprocessed_planning_problem_set
