from typing import Optional

import networkx as nx

from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class LongestPathLengthFilter(ScenarioFilter):
    """
    Rejects scenarios where the longest path is outside the specified range.
    """

    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario_bundle.preprocessed_scenario.lanelet_network)
        traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()
        longest_path = nx.dag_longest_path(traffic_flow_graph)
        longest_path_length = len(longest_path)

        if self.min_length is not None and longest_path_length >= self.min_length:
            return True
        if self.max_length is not None and longest_path_length <= self.max_length:
            return True
        return False
