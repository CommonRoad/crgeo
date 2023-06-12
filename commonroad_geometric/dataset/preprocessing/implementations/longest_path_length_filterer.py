import itertools
from typing import Optional, Tuple, Iterable

import numpy as np
import networkx as nx
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph


class LongestPathLengthFilterer(BaseScenarioFilterer):

    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)
        traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()
        longest_path = nx.dag_longest_path(traffic_flow_graph)
        longest_path_length = len(longest_path)

        if self.min_length is not None and longest_path_length >= self.min_length:
            return True
        if self.max_length is not None and longest_path_length <= self.max_length:
            return True
        return False
