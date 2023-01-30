import itertools
from typing import Tuple, Iterable

from commonroad.scenario.scenario import Scenario
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class LaneletNetworkSizeFilterer(BaseScenarioFilterer):

    def __init__(self, size_threshold: int):
        self.size_threshold = size_threshold
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum road network size for a scenario
        if len(scenario.lanelet_network.lanelets) >= self.size_threshold:
            return True
        return False
