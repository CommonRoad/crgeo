import pytest

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors import FunctionalScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.log_exception_wrapper import LogExceptionWrapper


@pytest.fixture
def raise_exception_preprocessor():
    def raise_exception(scenario_bundle: ScenarioBundle):
        raise RuntimeError

    return FunctionalScenarioPreprocessor(scenario_processor=raise_exception)


def test_raise_exception_preprocessor(
    scenario_bundle: ScenarioBundle,
    raise_exception_preprocessor
):
    with pytest.raises(RuntimeError) as e_info:
        raise_exception_preprocessor(scenario_bundle)


def test_log_exception_wrapper_suppresses(
    scenario_bundle: ScenarioBundle,
):
    log_exception_wrapper = LogExceptionWrapper(
        wrapped_preprocessor=IdentityPreprocessor()
    )
    results = log_exception_wrapper(scenario_bundle)
    assert results
    assert results == [scenario_bundle]


def test_log_exception_wrapper_suppresses_on_raise(
    scenario_bundle: ScenarioBundle,
    raise_exception_preprocessor
):
    log_exception_wrapper = LogExceptionWrapper(
        wrapped_preprocessor=raise_exception_preprocessor
    )
    results = log_exception_wrapper(scenario_bundle)
    assert not results
