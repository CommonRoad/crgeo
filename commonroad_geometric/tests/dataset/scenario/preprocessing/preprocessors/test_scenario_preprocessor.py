import pytest
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.tests.dataset.scenario.preprocessing.preprocessor_mocks import AddLaneletPreprocessor, RemoveLaneletsPreprocessor


def get_preprocessed_scenario(result: T_ScenarioPreprocessorResult) -> Scenario:
    assert result
    assert len(result) == 1
    scenario_bundle = result[0]
    return scenario_bundle.preprocessed_scenario


def test_remove_lanelets_preprocessor(scenario_bundle):
    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    result_singleton = remove_lanelets_preprocessor(scenario_bundle)
    preprocessed_scenario = get_preprocessed_scenario(result_singleton)
    assert not preprocessed_scenario.lanelet_network.lanelets
    assert not preprocessed_scenario.lanelet_network.lanelet_polygons


def test_add_lanelet_preprocessor(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet = mock_lanelet(lanelet_id_start)
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    result_singleton = add_lanelet_preprocessor(scenario_bundle)
    preprocessed_scenario = get_preprocessed_scenario(result_singleton)

    assert preprocessed_scenario.lanelet_network.lanelets
    assert len(preprocessed_scenario.lanelet_network.lanelets) == lanelet_count + 1
    assert lanelet in preprocessed_scenario.lanelet_network.lanelets


def test_chain_remove_lanelets_pp_add_lanelet_pp(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet = mock_lanelet(lanelet_id_start)
    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    chained_preprocessor = remove_lanelets_preprocessor >> add_lanelet_preprocessor

    result_singleton = chained_preprocessor(scenario_bundle)
    preprocessed_scenario = get_preprocessed_scenario(result_singleton)

    assert preprocessed_scenario.lanelet_network.lanelets
    assert len(preprocessed_scenario.lanelet_network.lanelets) == 1
    assert lanelet in preprocessed_scenario.lanelet_network.lanelets


def test_chain_add_lanelet_pp_remove_lanelets_pp(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet = mock_lanelet(lanelet_id_start)
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    chained_preprocessor = add_lanelet_preprocessor >> remove_lanelets_preprocessor

    result_singleton = chained_preprocessor(scenario_bundle)
    preprocessed_scenario = get_preprocessed_scenario(result_singleton)

    assert not preprocessed_scenario.lanelet_network.lanelets


@pytest.mark.parametrize("number_of_add_pp", range(2, 10))
def test_chain_n_times(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start,
    number_of_add_pp: int
):
    lanelets = []
    chained_preprocessor = RemoveLaneletsPreprocessor()
    for x in range(number_of_add_pp):
        lanelet = mock_lanelet(lanelet_id_start + x)
        lanelets.append(lanelet)
        add_pp = AddLaneletPreprocessor(lanelet)
        chained_preprocessor >>= add_pp

    result_singleton = chained_preprocessor(scenario_bundle)
    preprocessed_scenario = get_preprocessed_scenario(result_singleton)

    assert preprocessed_scenario.lanelet_network.lanelets
    assert len(preprocessed_scenario.lanelet_network.lanelets) == number_of_add_pp
    assert lanelets == preprocessed_scenario.lanelet_network.lanelets


def test_fail_chain_preprocessors_with_list(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start,
):
    add_preprocessors = [
        AddLaneletPreprocessor(mock_lanelet(lanelet_id_start)),
        AddLaneletPreprocessor(mock_lanelet(lanelet_id_start + 1))
    ]
    chained_preprocessor = RemoveLaneletsPreprocessor()
    # Chaining with a list should fail for now
    with pytest.raises(AttributeError) as e_info:
        chained_preprocessor >>= add_preprocessors


def test_multi_scenario_preprocessor_remove_pp_add_pp(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)
    # FIRST
    lanelet = mock_lanelet(lanelet_id_start)
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    # SECOND
    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    # FIRST | SECOND
    multi_preprocessor = add_lanelet_preprocessor | remove_lanelets_preprocessor

    results = multi_preprocessor(scenario_bundle)
    add_pp_scenario = results[0].preprocessed_scenario  # FIRST
    remove_pp_scenario = results[1].preprocessed_scenario  # SECOND

    assert not remove_pp_scenario.lanelet_network.lanelets

    assert add_pp_scenario.lanelet_network.lanelets
    assert len(add_pp_scenario.lanelet_network.lanelets) == lanelet_count + 1
    assert lanelet in add_pp_scenario.lanelet_network.lanelets


def test_multi_scenario_preprocessor_add_pp_remove_pp_order(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)
    # FIRST
    remove_lanelets_preprocessor = RemoveLaneletsPreprocessor()
    # SECOND
    lanelet = mock_lanelet(lanelet_id_start)
    add_lanelet_preprocessor = AddLaneletPreprocessor(lanelet)
    # FIRST | SECOND
    multi_preprocessor = remove_lanelets_preprocessor | add_lanelet_preprocessor

    results = multi_preprocessor(scenario_bundle)
    remove_pp_scenario = results[0].preprocessed_scenario  # FIRST
    add_pp_scenario = results[1].preprocessed_scenario  # SECOND

    assert add_pp_scenario.lanelet_network.lanelets
    assert len(add_pp_scenario.lanelet_network.lanelets) == lanelet_count + 1
    assert lanelet in add_pp_scenario.lanelet_network.lanelets

    assert not remove_pp_scenario.lanelet_network.lanelets


@pytest.mark.parametrize("number_of_remove_pp", range(1, 5))
def test_multi_preprocessing_same_pp_n_times(
    scenario_bundle,
    number_of_remove_pp: int
):
    multi_preprocessor = RemoveLaneletsPreprocessor()
    for _ in range(number_of_remove_pp):
        multi_preprocessor |= RemoveLaneletsPreprocessor()

    results = multi_preprocessor(scenario_bundle)

    for result_bundle in results:
        remove_pp_scenario = result_bundle.preprocessed_scenario
        assert not remove_pp_scenario.lanelet_network.lanelets


@pytest.mark.parametrize("number_of_remove_pp", range(1, 5))
def test_mul_equals_or_for_multi_preprocessing_same_pp_n_times(
    scenario_bundle,
    number_of_remove_pp: int
):
    or_multi_preprocessor = RemoveLaneletsPreprocessor()
    for _ in range(number_of_remove_pp - 1):  # We have to initialize the first one outside the loop
        or_multi_preprocessor |= RemoveLaneletsPreprocessor()

    mul_multi_preprocessor = RemoveLaneletsPreprocessor() * number_of_remove_pp

    or_results = or_multi_preprocessor(scenario_bundle)
    mul_results = mul_multi_preprocessor(scenario_bundle)

    assert len(or_results) == len(mul_results)
    assert or_results == mul_results
    for or_result, mul_result in zip(or_results, mul_results):
        or_scenario = or_result.preprocessed_scenario
        mul_scenario = mul_result.preprocessed_scenario

        assert not or_scenario.lanelet_network.lanelets
        assert not mul_scenario.lanelet_network.lanelets


@pytest.mark.parametrize("number_of_add_pp", range(1, 5))
def test_multi_preprocessing_different_pp_n_times(
    scenario_bundle,
    mock_lanelet,
    lanelet_id_start,
    number_of_add_pp: int
):
    lanelet_count = len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets)

    lanelets = []
    multi_preprocessor = RemoveLaneletsPreprocessor()
    for x in range(number_of_add_pp):
        lanelet = mock_lanelet(lanelet_id_start + x)
        lanelets.append(lanelet)
        add_pp = AddLaneletPreprocessor(lanelet)
        multi_preprocessor |= add_pp

    results = multi_preprocessor(scenario_bundle)
    remove_pp_scenario = results[0].preprocessed_scenario  # First preprocessor is RemoveLaneletsPreprocessor

    assert not remove_pp_scenario.lanelet_network.lanelets

    for result_bundle, lanelet in zip(results[1:], lanelets):
        add_pp_scenario = result_bundle.preprocessed_scenario
        assert add_pp_scenario.lanelet_network.lanelets
        assert len(add_pp_scenario.lanelet_network.lanelets) == lanelet_count + 1
        assert lanelet in add_pp_scenario.lanelet_network.lanelets
