"""
This file tests the use of preprocessing in combination with the ScenarioIterator.
We explicitly test this in  an additional file because this integrates two
components and because we also have to explicitly test for side effects within
the preprocessors which could be caused by the multiprocessing in the
ScenarioIterator.
"""

import pytest

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.tests.dataset.scenario.preprocessing.preprocessor_mocks import AcceptNFilter, AddLaneletPreprocessor, RemoveLaneletsPreprocessor, SafeAcceptNFilter


@pytest.fixture(scope="function")
def simple_multi_preprocessor(
    mock_lanelet,
    lanelet_id_start,
):
    lanelet1 = mock_lanelet(lanelet_id_start)
    add_lanelet_preprocessor1 = AddLaneletPreprocessor(lanelet1)

    lanelet2 = mock_lanelet(lanelet_id_start + 1)
    add_lanelet_preprocessor2 = AddLaneletPreprocessor(lanelet2)

    multi_preprocessor = add_lanelet_preprocessor1 | add_lanelet_preprocessor2
    return RemoveLaneletsPreprocessor() >> multi_preprocessor


@pytest.fixture(scope="function")
def unsafe_preprocessor(simple_multi_preprocessor):
    """
    This preprocessor suffers from race conditions and could (and will) accept
    more than one scenario when used in a multiprocessing context

                                 |-> add_lanelet_preprocessor1 -|
    remove_lanelets_preprocessor-|                              |-> accept_n_filter
                                 |-> add_lanelet_preprocessor2 -|
    """
    accept_n_filter = AcceptNFilter(n=1)
    preprocessor = simple_multi_preprocessor >> accept_n_filter
    return preprocessor


@pytest.fixture(scope="function")
def safe_preprocessor(simple_multi_preprocessor):
    """
    This preprocessor does not suffer from any race conditions and will accept
    exactly one scenario every time, even when used in a multiprocessing context
    as SafeAcceptNFilter uses a lock shared between processes

                                 |-> add_lanelet_preprocessor1 -|
    remove_lanelets_preprocessor-|                              |-> safe_accept_n_filter
                                 |-> add_lanelet_preprocessor2 -|
    """
    safe_accept_n_filter = SafeAcceptNFilter(n=1)
    preprocessor = simple_multi_preprocessor >> safe_accept_n_filter
    return preprocessor


def test_sync_scenario_iterator_remove_lanelets_preprocessor(scenario_directory_xml):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=RemoveLaneletsPreprocessor(),
        workers=1
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None
        assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelets
        assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelet_polygons


def test_async_scenario_iterator_remove_lanelets_preprocessor(scenario_directory_xml):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=RemoveLaneletsPreprocessor(),
        workers=2
    )

    for scenario_bundle in scenario_iterator:
        assert scenario_bundle is not None
        assert type(scenario_bundle) is ScenarioBundle
        assert scenario_bundle.scenario_path is not None
        assert scenario_bundle.input_scenario is not None
        assert scenario_bundle.preprocessed_scenario is not None
        assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelets
        assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelet_polygons


def test_sync_scenario_iterator_unsafe_preprocessor_has_no_race_condition(
    scenario_directory_xml,
    unsafe_preprocessor,
    lanelet_id_start
):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=unsafe_preprocessor,
        workers=1
    )
    results = [scenario_bundle for scenario_bundle in scenario_iterator]

    assert unsafe_preprocessor.results_factor == 2
    assert len(results) == 1

    first_result = results[0]

    first_lanelet_network = first_result.preprocessed_scenario.lanelet_network
    assert first_lanelet_network.lanelets
    assert len(first_lanelet_network.lanelets) == 1
    first_lanelet = first_lanelet_network.lanelets[0]
    assert first_lanelet.lanelet_id == lanelet_id_start


def test_async_scenario_iterator_unsafe_preprocessor_has_race_condition(
    scenario_directory_xml,
    unsafe_preprocessor,
    lanelet_id_start,
):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=unsafe_preprocessor,
        workers=4
    )
    results = [scenario_bundle for scenario_bundle in scenario_iterator]

    assert unsafe_preprocessor.results_factor == 2
    assert len(results) != 1

    # Making sure that the only side effect of the race condition is too many results,
    # and no further issues occur as the remaining preprocessor is stateless/safe
    for scenario_bundle in results:
        lanelet_network = scenario_bundle.preprocessed_scenario.lanelet_network
        assert lanelet_network.lanelets
        assert len(lanelet_network.lanelets) == 1
        lanelet = lanelet_network.lanelets[0]
        assert lanelet.lanelet_id in [lanelet_id_start, lanelet_id_start + 1]


def test_sync_scenario_iterator_safe_preprocessor_has_no_race_condition(
    scenario_directory_xml,
    safe_preprocessor,
    lanelet_id_start
):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=safe_preprocessor,
        workers=1
    )
    results = [scenario_bundle for scenario_bundle in scenario_iterator]

    assert safe_preprocessor.results_factor == 2
    assert len(results) == 1

    first_result = results[0]

    first_lanelet_network = first_result.preprocessed_scenario.lanelet_network
    assert first_lanelet_network.lanelets
    assert len(first_lanelet_network.lanelets) == 1
    first_lanelet = first_lanelet_network.lanelets[0]
    assert first_lanelet.lanelet_id == lanelet_id_start


def test_async_scenario_iterator_safe_preprocessor_has_no_race_condition(
    scenario_directory_xml,
    safe_preprocessor,
    lanelet_id_start,
):
    scenario_iterator = ScenarioIterator(
        directory=scenario_directory_xml,
        preprocessor=safe_preprocessor,
        workers=4
    )
    results = [scenario_bundle for scenario_bundle in scenario_iterator]

    assert safe_preprocessor.results_factor == 2
    assert len(results) == 1

    first_result = results[0]

    first_lanelet_network = first_result.preprocessed_scenario.lanelet_network
    assert first_lanelet_network.lanelets
    assert len(first_lanelet_network.lanelets) == 1
    first_lanelet = first_lanelet_network.lanelets[0]
    assert first_lanelet.lanelet_id == lanelet_id_start
