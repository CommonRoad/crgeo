import shutil
from pathlib import Path

import pytest
from commonroad.common.file_writer import OverwriteExistingFile

from commonroad_geometric.common.io_extensions.scenario_file_format import ScenarioFileFormat
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.scenario_format_converter import ScenarioFormatConverter


@pytest.fixture
def output_directory():
    return Path('./output', Path(__file__).stem)


@pytest.mark.parametrize(
    argnames='file_format',
    argvalues=[
        ScenarioFileFormat.XML,
        ScenarioFileFormat.BUNDLE,
        ScenarioFileFormat.ALL,
    ]
)
def test_convert_to_file_format(
    scenario_bundle: ScenarioBundle,
    output_directory: Path,
    file_format: ScenarioFileFormat,
):
    # Delete base output directory before every test to prevent side effects
    shutil.rmtree(output_directory, ignore_errors=True)
    converter = ScenarioFormatConverter(
        output_file_format=file_format,
        output_directory=output_directory,
        decimal_precision=10,
        overwrite_existing_file=OverwriteExistingFile.ALWAYS
    )

    path = scenario_bundle.scenario_path
    expected_output_paths = [
        output_directory / (path.stem + suffix)
        for suffix in file_format.suffixes
    ]

    # Delete files at expected output path before test
    for expected_output_path in expected_output_paths:
        expected_output_path.unlink(missing_ok=True)
        assert not expected_output_path.is_file()

    result = converter(scenario_bundle)
    for expected_output_path in expected_output_paths:
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
        assert (saved_scenario_bundle.preprocessed_planning_problem_set ==
                scenario_bundle.preprocessed_planning_problem_set)


@pytest.mark.parametrize(
    argnames='file_format',
    argvalues=[
        ScenarioFileFormat.XML,
        ScenarioFileFormat.BUNDLE,
        ScenarioFileFormat.ALL,
    ]
)
def test_convert_no_output_directory(
    scenario_bundle: ScenarioBundle,
    output_directory: Path,
    file_format: ScenarioFileFormat,
):
    # Delete output directory before every test to prevent side effects
    shutil.rmtree(output_directory, ignore_errors=True)
    converter = ScenarioFormatConverter(
        output_file_format=file_format,
        decimal_precision=10,
        overwrite_existing_file=OverwriteExistingFile.ALWAYS
    )

    path = scenario_bundle.scenario_path
    expected_output_paths = [
        output_directory / (path.stem + suffix)
        for suffix in file_format.suffixes
    ]

    # Delete files at expected output path before test
    for expected_output_path in expected_output_paths:
        expected_output_path.unlink(missing_ok=True)
        assert not expected_output_path.is_file()

    # Modify scenario_path to pretend that this scenario bundle comes from the
    # expected output directory, could be replaced by path mock
    scenario_bundle.scenario_path = output_directory / path.name
    result = converter(scenario_bundle)
    # Revert path as to not modify original bundle
    scenario_bundle.scenario_path = path
    for expected_output_path in expected_output_paths:
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
        assert (saved_scenario_bundle.preprocessed_planning_problem_set ==
                scenario_bundle.preprocessed_planning_problem_set)
