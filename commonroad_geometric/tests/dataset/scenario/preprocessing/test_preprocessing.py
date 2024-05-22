from copy import deepcopy

import pytest
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.tests.dataset.scenario.preprocessing.preprocessor_mocks import AcceptNFilter, AddLaneletPreprocessor, RemoveLaneletsPreprocessor, _MaxLaneletCountFilter, \
    _MinLaneletCountFilter


def get_preprocessed_scenario(result: T_ScenarioPreprocessorResult) -> Scenario:
    assert result
    assert len(result) == 1
    scenario_bundle = result[0]
    return scenario_bundle.preprocessed_scenario


def test_remove_lanelets_filter_min_size(scenario_bundle):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    min_lanelet_count_filter = _MinLaneletCountFilter(min_size_threshold=lanelet_count)
    remove_then_filter = remove_lanelets_preprocessor >> min_lanelet_count_filter

    empty_results = remove_then_filter(scenario_bundle)

    assert empty_results == []


def test_remove_lanelets_inverted_filter_min_size(scenario_bundle):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    min_lanelet_count_filter = _MinLaneletCountFilter(min_size_threshold=lanelet_count)
    remove_then_filter = remove_lanelets_preprocessor >> ~min_lanelet_count_filter

    removed_lanelets_results = remove_then_filter(scenario_bundle)

    assert removed_lanelets_results == [scenario_bundle]
    assert scenario_bundle.input_scenario.lanelet_network.lanelets
    assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelets


def test_filter_min_size_remove_lanelets(scenario_bundle):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    min_lanelet_count_filter = _MinLaneletCountFilter(min_size_threshold=lanelet_count)
    filter_then_remove = min_lanelet_count_filter >> remove_lanelets_preprocessor

    removed_lanelets_results = filter_then_remove(scenario_bundle)

    assert removed_lanelets_results == [scenario_bundle]
    assert scenario_bundle.input_scenario.lanelet_network.lanelets
    assert not scenario_bundle.preprocessed_scenario.lanelet_network.lanelets


def test_add_lanelet_filter_max_size(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    max_lanelet_count_filter = _MaxLaneletCountFilter(max_size_threshold=lanelet_count)
    add_then_filter = add_lanelet_preprocessor >> max_lanelet_count_filter

    empty_results = add_then_filter(scenario_bundle)

    assert empty_results == []


def test_filter_max_size_add_lanelet(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    max_lanelet_count_filter = _MaxLaneletCountFilter(max_size_threshold=lanelet_count)
    filter_then_add = max_lanelet_count_filter >> add_lanelet_preprocessor

    add_lanelet_results = filter_then_add(scenario_bundle)

    assert add_lanelet_results == [scenario_bundle]
    assert len(scenario_bundle.input_scenario.lanelet_network.lanelets) == lanelet_count
    assert len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) == lanelet_count + 1


def test_remove_lanelets_filter_max_add_lanelet_filter_min(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start,
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    remove_filter_max_add_filter_min = RemoveLaneletsPreprocessor()
    remove_filter_max_add_filter_min >>= _MaxLaneletCountFilter(max_size_threshold=lanelet_count // 2)
    remove_filter_max_add_filter_min >>= AddLaneletPreprocessor(lanelet)
    remove_filter_max_add_filter_min >>= _MinLaneletCountFilter(min_size_threshold=1)

    results = remove_filter_max_add_filter_min(scenario_bundle)
    assert results == [scenario_bundle]
    assert len(scenario_bundle.input_scenario.lanelet_network.lanelets) == lanelet_count
    assert len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) == 1


def test_chain_preprocessors_util(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    remove_filter_max_add_filter_min = chain_preprocessors(
        RemoveLaneletsPreprocessor(),
        _MaxLaneletCountFilter(max_size_threshold=lanelet_count // 2),
        AddLaneletPreprocessor(lanelet),
        _MinLaneletCountFilter(min_size_threshold=1)
    )

    results = remove_filter_max_add_filter_min(scenario_bundle)
    assert results == [scenario_bundle]
    assert len(scenario_bundle.input_scenario.lanelet_network.lanelets) == lanelet_count
    assert len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) == 1


def test_multi_preprocessor_filter_min_size(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    min_lanelet_count_filter = _MinLaneletCountFilter(min_size_threshold=1)
    # FIRST | SECOND
    multi_preprocessor = remove_lanelets_preprocessor | add_lanelet_preprocessor
    multi_preprocessor >>= min_lanelet_count_filter

    results = multi_preprocessor(scenario_bundle)
    # Should be unequal and different objects
    assert results != [scenario_bundle]
    assert results is not [scenario_bundle]

    preprocessed_bundle = results[0]
    assert len(preprocessed_bundle.input_scenario.lanelet_network.lanelets) == lanelet_count

    assert preprocessed_bundle.preprocessed_scenario.lanelet_network.lanelets
    assert len(preprocessed_bundle.preprocessed_scenario.lanelet_network.lanelets) == lanelet_count + 1


def test_multi_preprocessor_and_filters(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelet = mock_lanelet(lanelet_id_start)

    remove_all_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    remove_half_lanelets_preprocessor = RemoveLaneletsPreprocessor(removed_lanelet_count=lanelet_count // 2)
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    min_lanelet_count_filter = _MinLaneletCountFilter(min_size_threshold=1)
    max_lanelet_count_filter = _MaxLaneletCountFilter(max_size_threshold=(lanelet_count // 2) + 1)
    # FIRST | SECOND
    multi_preprocessor = remove_half_lanelets_preprocessor | (remove_all_lanelets_preprocessor >> add_lanelet_preprocessor)
    multi_preprocessor >>= min_lanelet_count_filter & max_lanelet_count_filter

    results = multi_preprocessor(scenario_bundle)

    assert results
    assert len(results) == 2
    half_removed_result = results[0]
    one_lanelet_result = results[1]

    assert len(half_removed_result.input_scenario.lanelet_network.lanelets) == lanelet_count

    assert half_removed_result.preprocessed_scenario.lanelet_network.lanelets
    assert len(half_removed_result.preprocessed_scenario.lanelet_network.lanelets) == lanelet_count - lanelet_count // 2

    assert len(one_lanelet_result.input_scenario.lanelet_network.lanelets) == lanelet_count

    assert one_lanelet_result.preprocessed_scenario.lanelet_network.lanelets
    assert len(one_lanelet_result.preprocessed_scenario.lanelet_network.lanelets) == 1


def test_multi_preprocessing_chained_preprocessor_possible():
    # This shouldn't raise an exception
    RemoveLaneletsPreprocessor() >> AcceptNFilter(n=1) | RemoveLaneletsPreprocessor() >> RemoveLaneletsPreprocessor()
    RemoveLaneletsPreprocessor() >> AcceptNFilter(n=1) | RemoveLaneletsPreprocessor() >> AcceptNFilter(n=1)


def test_multi_preprocessing_has_no_race_condition(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    """
    This preprocessor shouldn't suffer from any race conditions and accept the
    first scenario from add_lanelet_preprocessor1 and reject the second scenario
    from add_lanelet_preprocessor2

                                 |-> add_lanelet_preprocessor1 -|
    remove_lanelets_preprocessor-|                              |-> accept_n_filter
                                 |-> add_lanelet_preprocessor2 -|
    """
    lanelet1 = mock_lanelet(lanelet_id_start)
    # FIRST
    add_lanelet_preprocessor1 = AddLaneletPreprocessor(lanelet1)

    lanelet2 = mock_lanelet(lanelet_id_start + 1)
    # SECOND
    add_lanelet_preprocessor2 = AddLaneletPreprocessor(lanelet2)

    # FIRST | SECOND
    multi_preprocessor = add_lanelet_preprocessor1 | add_lanelet_preprocessor2

    preprocessor = RemoveLaneletsPreprocessor() >> multi_preprocessor >> AcceptNFilter(n=1)
    results = preprocessor(scenario_bundle)

    assert preprocessor.results_factor == 2
    assert len(results) == 1

    first_result = results[0]

    first_lanelet_network = first_result.preprocessed_scenario.lanelet_network
    assert first_lanelet_network.lanelets
    assert len(first_lanelet_network.lanelets) == 1
    first_lanelet = first_lanelet_network.lanelets[0]
    assert first_lanelet.lanelet_id == lanelet_id_start


def test_multi_preprocessing_copy_has_no_side_effects(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    """
    This preprocessor shouldn't suffer from any side effects and accept the
    first scenario from add_lanelet_preprocessor1 and accept the second scenario
    from add_lanelet_preprocessor2

                                 |-> add_lanelet_preprocessor1 -> accept_n_filter
    remove_lanelets_preprocessor-|
                                 |-> add_lanelet_preprocessor2 -> accept_n_filter_copy
    """
    lanelet1 = mock_lanelet(lanelet_id_start)
    # FIRST
    add_lanelet_preprocessor1 = AddLaneletPreprocessor(lanelet1)

    lanelet2 = mock_lanelet(lanelet_id_start + 1)
    # SECOND
    add_lanelet_preprocessor2 = AddLaneletPreprocessor(lanelet2)

    # FIRST | SECOND
    accept_n_filter = AcceptNFilter(n=1)
    accept_n_filter_copy = deepcopy(accept_n_filter)
    # >> has higher precedence than |
    multi_preprocessor = add_lanelet_preprocessor1 >> accept_n_filter | add_lanelet_preprocessor2 >> accept_n_filter_copy

    preprocessor = RemoveLaneletsPreprocessor() >> multi_preprocessor
    results = preprocessor(scenario_bundle)

    assert preprocessor.results_factor == 2
    assert len(results) == 2

    first_result = results[0]
    second_result = results[1]

    first_lanelet_network = first_result.preprocessed_scenario.lanelet_network
    assert first_lanelet_network.lanelets
    assert len(first_lanelet_network.lanelets) == 1
    first_lanelet = first_lanelet_network.lanelets[0]
    assert first_lanelet.lanelet_id == lanelet_id_start

    second_lanelet_network = second_result.preprocessed_scenario.lanelet_network
    assert second_lanelet_network.lanelets
    assert len(second_lanelet_network.lanelets) == 1
    second_lanelet = second_lanelet_network.lanelets[0]
    assert second_lanelet.lanelet_id == lanelet_id_start + 1
