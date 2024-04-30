from __future__ import annotations

import networkx as nx
from networkx.exception import NetworkXNoCycle

from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class TrafficFlowCycleFilter(ScenarioFilter):
    """
    Rejects scenarios with cycles in the road network/traffic flow graph.

    """

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario_bundle.preprocessed_scenario.lanelet_network)
        traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()

        try:
            nx.find_cycle(traffic_flow_graph)
            has_cycle = True
        except NetworkXNoCycle:
            has_cycle = False

        return not has_cycle
