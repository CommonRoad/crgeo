from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, List
import networkx as nx
from networkx.exception import NetworkXNoCycle
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph

class CycleFilterer(BaseScenarioFilterer):
    """
    Removes or keeps scenarios with cycles in the road network.
    """

    def __init__(
        self,
        keep_cycles: bool = False
    ) -> None:
        self.keep_cycles = keep_cycles
        super(CycleFilterer, self).__init__()


    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario

        lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)
        traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()
        
        try:
            nx.find_cycle(traffic_flow_graph)
            has_cycle = True
        except NetworkXNoCycle:
            has_cycle = False

        if has_cycle and self.keep_cycles:
            return True
        if not has_cycle and not self.keep_cycles:
            return True
        return False
