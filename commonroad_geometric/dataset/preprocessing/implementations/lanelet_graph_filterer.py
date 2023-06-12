import itertools
from typing import Optional, Tuple, Iterable

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph


class LaneletGraphFilterer(BaseScenarioFilterer):

    def __init__(
        self, 
        min_nodes: Optional[int] = None,
        max_nodes: Optional[int] = None,
        min_edges: Optional[int] = None,
        max_edges: Optional[int] = None,
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)
        
        num_nodes = lanelet_graph.number_of_nodes()
        num_edges = lanelet_graph.number_of_edges()

        if self.min_nodes is not None and num_nodes < self.min_nodes:
            return False
        if self.max_nodes is not None and num_nodes > self.max_nodes:
            return False
        if self.min_edges is not None and num_edges < self.min_edges:
            return False
        if self.max_edges is not None and num_edges > self.max_edges:
            return False

        return True
