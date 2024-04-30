from typing import Callable

import numpy as np
import pytest
from commonroad.scenario.lanelet import Lanelet

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle


@pytest.fixture(scope="function")
def scenario_bundle(arg_carcarana_path_xml) -> ScenarioBundle:
    # Reading the file once is not sufficient to test the ScenarioPreprocessor, as it could modify the scenario
    return ScenarioBundle.read(scenario_path=arg_carcarana_path_xml)


@pytest.fixture(scope="class")
def mock_lanelet() -> Callable[[int], Lanelet]:
    def make_lanelet(lanelet_id: int) -> Lanelet:
        return Lanelet(
            right_vertices=np.array([[0, 0], [1, 0], [2, 0]]),
            left_vertices=np.array([[0, 1], [1, 1], [2, 1]]),
            center_vertices=np.array([[0, .5], [1, .5], [2, .5]]),
            lanelet_id=lanelet_id,
        )

    return make_lanelet


@pytest.fixture(scope="function")
def lanelet_id_start() -> int:
    # Should be high enough to avoid scenario's IDs, sufficient many for all tests
    return 100_000
