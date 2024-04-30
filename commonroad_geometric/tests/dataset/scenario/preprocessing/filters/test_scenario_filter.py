from typing import Sequence

import pytest

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import FunctionalScenarioFilter, ScenarioFilter


@pytest.fixture(scope="function")
def expected_accepted_result(scenario_bundle) -> Sequence[ScenarioBundle]:
    return [scenario_bundle]


# Simple filters for tests
class RejectFilter(ScenarioFilter):
    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return False


class AcceptFilter(ScenarioFilter):
    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return True


def test_reject_filter(scenario_bundle):
    reject_filter = RejectFilter()
    assert reject_filter(scenario_bundle) == []


def test_accept_filter(scenario_bundle, expected_accepted_result):
    accept_filter = AcceptFilter()
    assert accept_filter(scenario_bundle) == expected_accepted_result


def test_functional_accept_filter(scenario_bundle, expected_accepted_result):
    functional_accept_filter = FunctionalScenarioFilter(filter_callable=lambda bundle: True)
    assert functional_accept_filter(scenario_bundle) == expected_accepted_result


def test_reject_filter_equals_inverted_accept_filter(scenario_bundle):
    reject_filter = RejectFilter()
    accept_filter = AcceptFilter()
    inverted_accept_filter = ~accept_filter
    assert reject_filter(scenario_bundle) == inverted_accept_filter(scenario_bundle)


def test_chain_reject_filters(scenario_bundle):
    chained_filter = RejectFilter() >> RejectFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_accept_filters(scenario_bundle, expected_accepted_result):
    chained_filter = AcceptFilter() >> AcceptFilter()
    assert chained_filter(scenario_bundle) == expected_accepted_result


def test_chain_reject_filter_accept_filter(scenario_bundle):
    chained_filter = RejectFilter() >> AcceptFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_accept_filter_reject_filters(scenario_bundle):
    chained_filter = AcceptFilter() >> RejectFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_assignment_reject_filters(scenario_bundle):
    chained_filter = RejectFilter()
    chained_filter >>= RejectFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_assignment_accept_filters(scenario_bundle, expected_accepted_result):
    chained_filter = AcceptFilter()
    chained_filter >>= AcceptFilter()
    assert chained_filter(scenario_bundle) == expected_accepted_result


def test_chain_assignment_reject_filter_accept_filter(scenario_bundle):
    chained_filter = RejectFilter()
    chained_filter >>= AcceptFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_assignment_accept_filter_reject_filters(scenario_bundle):
    chained_filter = AcceptFilter()
    chained_filter >>= RejectFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_inverted_reject_filters(scenario_bundle, expected_accepted_result):
    chained_filter = ~RejectFilter() >> ~RejectFilter()
    assert chained_filter(scenario_bundle) == expected_accepted_result


def test_chain_inverted_accept_filters(scenario_bundle):
    chained_filter = ~AcceptFilter() >> ~AcceptFilter()
    assert chained_filter(scenario_bundle) == []


def test_chain_inverted_reject_filter_accept_filter(scenario_bundle, expected_accepted_result):
    chained_filter = ~RejectFilter() >> AcceptFilter()
    assert chained_filter(scenario_bundle) == expected_accepted_result


def test_chain_inverted_accept_filter_reject_filters(scenario_bundle):
    chained_filter = ~AcceptFilter() >> RejectFilter()
    assert chained_filter(scenario_bundle) == []


def test_and_equals_mul_for_filters(scenario_bundle):
    and_filter = AcceptFilter() & RejectFilter()
    chained_filter = AcceptFilter() >> RejectFilter()
    assert chained_filter(scenario_bundle) == and_filter(scenario_bundle)


def test_and_assignment_accept_filters(scenario_bundle):
    and_filter = AcceptFilter()
    and_filter &= RejectFilter()
    assert and_filter(scenario_bundle) == []


def test_or_accept_filter_reject_filter(scenario_bundle, expected_accepted_result):
    or_filter = AcceptFilter() | RejectFilter()
    assert or_filter(scenario_bundle) == expected_accepted_result


def test_or_reject_filter_reject_filter(scenario_bundle):
    or_filter = RejectFilter() | RejectFilter()
    assert or_filter(scenario_bundle) == []


def test_or_accept_filter_accept_filter(scenario_bundle, expected_accepted_result):
    or_filter = AcceptFilter() | AcceptFilter()
    assert or_filter(scenario_bundle) == expected_accepted_result


def test_or_assignment_accept_filter_reject_filter(scenario_bundle, expected_accepted_result):
    or_filter = AcceptFilter()
    or_filter |= RejectFilter()
    assert or_filter(scenario_bundle) == expected_accepted_result


def test_xor_accept_filter_reject_filter(scenario_bundle, expected_accepted_result):
    xor_filter = AcceptFilter() ^ RejectFilter()
    assert xor_filter(scenario_bundle) == expected_accepted_result


def test_xor_reject_filter_accept_filter(scenario_bundle, expected_accepted_result):
    xor_filter = RejectFilter() ^ AcceptFilter()
    assert xor_filter(scenario_bundle) == expected_accepted_result


def test_xor_reject_filter_reject_filter(scenario_bundle):
    xor_filter = RejectFilter() ^ RejectFilter()
    assert xor_filter(scenario_bundle) == []


def test_xor_accept_filter_accept_filter(scenario_bundle):
    xor_filter = AcceptFilter() ^ AcceptFilter()
    assert xor_filter(scenario_bundle) == []


def test_xor_assignment_accept_filter_reject_filter(scenario_bundle, expected_accepted_result):
    xor_filter = AcceptFilter()
    xor_filter ^= RejectFilter()
    assert xor_filter(scenario_bundle) == expected_accepted_result
