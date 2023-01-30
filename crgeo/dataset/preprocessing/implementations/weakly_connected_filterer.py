from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, List
import networkx as nx
from networkx.exception import NetworkXNoCycle
from commonroad.scenario.scenario import Scenario
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph

class WeaklyConnectedFilterer(BaseScenarioFilterer):
    """
    Removes or keeps scenarios with cycles in the road network.
    """

    def __init__(
        self,
        keep_cycles: bool = False
    ) -> None:
        self.keep_cycles = keep_cycles
        super(WeaklyConnectedFilterer, self).__init__()


    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario

        lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)
        #traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()
        
        return nx.is_weakly_connected(lanelet_graph)
