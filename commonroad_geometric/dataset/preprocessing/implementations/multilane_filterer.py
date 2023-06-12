import itertools
from typing import Tuple, Iterable

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class MultiLaneFilterer(BaseScenarioFilterer):

    def __init__(self, keep_multilane: bool = False):
        self.keep_multilane = keep_multilane
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario

        has_highway: bool = False
        
        for lanelet in scenario.lanelet_network.lanelets:
            if (lanelet.adj_left and lanelet.adj_left_same_direction) or \
                (lanelet.adj_right and lanelet.adj_right_same_direction):
                has_highway = True
                break 
        
        if has_highway and self.keep_multilane:
            return True
        if not has_highway and not self.keep_multilane:
            return True
        return False