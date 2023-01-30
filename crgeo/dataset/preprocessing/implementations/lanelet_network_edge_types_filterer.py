import itertools
from typing import Tuple, Iterable, Set

from typing import TYPE_CHECKING

import numpy as np
from crgeo.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad.scenario.scenario import Scenario
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class LaneletNetworkEdgeTypesFilterer(BaseScenarioFilterer):

    def __init__(
        self, 
        required_edges_types: Set[LaneletEdgeType]
    ):
        self.required_edges_types = required_edges_types
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        lanelet_graph = LaneletGraph.from_scenario(scenario)
        included_edge_types = set([LaneletEdgeType(e[2]['lanelet_edge_type']) for e in lanelet_graph.edges(data=True)])
        return len(self.required_edges_types - included_edge_types) == 0
