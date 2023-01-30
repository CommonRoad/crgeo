import itertools
from typing import Optional, Tuple, Iterable

from typing import TYPE_CHECKING

import numpy as np
from commonroad.scenario.scenario import Scenario
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class LaneletLengthFilterer(BaseScenarioFilterer):

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        min_length_scenario = min((l.distance[-1] for l in scenario.lanelet_network.lanelets))
        max_length_scenario = max((l.distance[-1] for l in scenario.lanelet_network.lanelets))

        if (self.min_length is None or min_length_scenario >= self.min_length) and \
           (self.max_length is None or max_length_scenario <= self.max_length):
           return True

        return False
