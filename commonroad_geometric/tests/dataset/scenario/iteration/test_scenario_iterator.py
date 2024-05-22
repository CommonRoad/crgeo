"""
This file tests the basic functionalities of the ScenarioIterator.
"""
from collections import defaultdict

from commonroad_geometric.common.io_extensions.scenario_file_format import ScenarioFileFormat
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator


def test_sync_scenario_iterator_file(arg_carcarana_path_xml):
    scenario_iterator = ScenarioIterator(
        directory=arg_carcarana_path_xml,
        workers=1
    )

    scenario_bundles = []
    for scenario_bundle in scenario_iterator:
        scenario_bundles.append(scenario_bundle)

        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.scenario_path.samefile(arg_carcarana_path_xml)
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None
        # File has planning_problem_set
        assert scenario_bundle.input_planning_problem_set is not None
        assert scenario_bundle.preprocessed_planning_problem_set is not None

    assert len(scenario_bundles) == 1


def test_sync_scenario_iterator(scenario_directory_xml):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        workers=1
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None


def test_async_scenario_iterator_file(arg_carcarana_path_xml):
    scenario_iterator = ScenarioIterator(
        directory=arg_carcarana_path_xml,
        workers=2
    )

    scenario_bundles = []
    for scenario_bundle in scenario_iterator:
        scenario_bundles.append(scenario_bundle)

        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.scenario_path.samefile(arg_carcarana_path_xml)
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None
        # File has planning_problem_set
        assert scenario_bundle.input_planning_problem_set is not None
        assert scenario_bundle.preprocessed_planning_problem_set is not None

    assert len(scenario_bundles) == 1


def test_async_scenario_iterator_xml(scenario_directory_xml):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        workers=2
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None


def test_async_scenario_iterator_pkl(scenario_directory_pickle):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_pickle,
        file_format=ScenarioFileFormat.BUNDLE,
        workers=2
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None


def test_async_scenario_iterator_all(root_testdata_path):
    scenario_iterator = ScenarioIterator(
        directory=root_testdata_path,
        file_format=ScenarioFileFormat.ALL,
        workers=2
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None


def test_scenario_iterator_sorted(scenario_directory_xml):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        is_looping=False,
        seed=None,
        workers=1  # Required for deterministic tests
    )

    scenario_paths = [scenario_bundle.scenario_path for scenario_bundle in scenario_iterator]
    sorted_scenario_paths = sorted(scenario_paths)

    assert sorted_scenario_paths == scenario_paths


def test_scenario_iterator_shuffled(scenario_directory_xml):
    shuffled_scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        is_looping=False,
        seed=42,
        workers=1,  # Required for deterministic tests
    )

    shuffled_scenario_paths = [scenario_bundle.scenario_path for scenario_bundle in shuffled_scenario_iterator]
    sorted_scenario_paths = sorted(shuffled_scenario_paths)

    assert sorted_scenario_paths != shuffled_scenario_paths


def test_scenario_iterator_loops_file(arg_carcarana_path_xml):
    looping_scenario_iterator = ScenarioIterator(
        directory=arg_carcarana_path_xml,
        is_looping=True,
        workers=1,
    )

    iterations = 5
    scenario_bundles = []
    for scenario_bundle, _ in zip(looping_scenario_iterator, range(iterations)):
        scenario_bundles.append(scenario_bundle)

        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.scenario_path.samefile(arg_carcarana_path_xml)
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None
        # File has planning_problem_set
        assert scenario_bundle.input_planning_problem_set is not None
        assert scenario_bundle.preprocessed_planning_problem_set is not None

    assert len(scenario_bundles) == iterations
    assert [scenario_bundles[0]] * 5 == scenario_bundles


def test_scenario_iterator_loops_sorted(scenario_directory_xml):
    looping_scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        is_looping=True,
        seed=None,
        workers=1,  # Required for deterministic tests
    )

    scenario_path_to_seen_count = defaultdict(int)
    scenario_paths = []
    for scenario_bundle in looping_scenario_iterator:
        path = scenario_bundle.scenario_path

        scenario_paths.append(path)
        scenario_path_to_seen_count[path] += 1

        if all([count > 1 for count in scenario_path_to_seen_count.values()]):
            break

    # With deterministic, sorted iteration we should have seen every scenario path exactly twice
    assert all(count == 2 for count in scenario_path_to_seen_count.values())

    # Sorting unique paths in scenario_path_to_seen_count
    sorted_scenario_paths = sorted(scenario_path_to_seen_count.keys())
    expected_scenario_paths = sorted_scenario_paths + sorted_scenario_paths

    assert len(scenario_paths) == len(expected_scenario_paths)
    assert scenario_paths == expected_scenario_paths


def test_scenario_iterator_loops_shuffled(scenario_directory_xml):
    looping_shuffled_scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        is_looping=True,
        seed=42,
        workers=1,  # Required for deterministic tests
    )

    scenario_path_to_seen_count = defaultdict(int)
    scenario_paths = []
    for scenario_bundle in looping_shuffled_scenario_iterator:
        path = scenario_bundle.scenario_path

        scenario_paths.append(path)
        scenario_path_to_seen_count[path] += 1

        if all([count > 1 for count in scenario_path_to_seen_count.values()]):
            break

    # With deterministic iteration we should have seen every scenario path exactly twice
    assert all(count == 2 for count in scenario_path_to_seen_count.values())

    # scenario_path_to_seen_count retains iteration order of first round through scenario paths
    sorted_scenario_paths = sorted(scenario_path_to_seen_count.keys())
    scenario_paths_as_seen = list(scenario_path_to_seen_count.keys())
    expected_scenario_paths = scenario_paths_as_seen + scenario_paths_as_seen

    assert len(scenario_paths) == len(expected_scenario_paths)
    assert sorted_scenario_paths != scenario_paths_as_seen
    assert scenario_paths == expected_scenario_paths


def test_async_scenario_iterator_loops(scenario_directory_xml):
    looping_async_scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        is_looping=True,
        seed=None,
        workers=4,
    )
    expected_scenario_paths = looping_async_scenario_iterator.scenario_paths

    scenario_path_to_seen_count = defaultdict(int)
    scenario_paths = []
    for scenario_bundle in looping_async_scenario_iterator:
        path = scenario_bundle.scenario_path

        scenario_paths.append(path)
        scenario_path_to_seen_count[path] += 1

        if all([count > 1 for count in scenario_path_to_seen_count.values()]):
            break

    # With non-deterministic iteration we can not guarantee that we have seen every scenario exactly twice
    # However we should make sure that we have seen every expected path at least twice
    for path in expected_scenario_paths:
        assert scenario_path_to_seen_count[path] >= 2

    scenario_paths_seen = set(scenario_path_to_seen_count.keys())

    assert len(scenario_paths) >= len(expected_scenario_paths) * 2
    assert scenario_paths_seen == set(expected_scenario_paths)
