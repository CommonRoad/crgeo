from __future__ import annotations

import networkx as nx

from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class WeaklyConnectedFilter(ScenarioFilter):
    """
    Accepts scenarios with weakly connected road networks.
    """

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario_bundle.preprocessed_scenario.lanelet_network)
        return nx.is_weakly_connected(lanelet_graph)
