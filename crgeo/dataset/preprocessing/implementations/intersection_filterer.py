import itertools
from typing import Optional, Tuple, Iterable

from typing import TYPE_CHECKING

import numpy as np
from commonroad.scenario.scenario import Scenario
from crgeo.common.io_extensions.lanelet_network import get_intersection_successors, remove_empty_intersections
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import compute_curvature
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer


class IntersectionFilterer(BaseScenarioFilterer):

    def __init__(
        self,
        min_intersections: Optional[int] = 1,
        max_intersections: Optional[int] = None,
        only_complete: bool = True
    ) -> None:
        self.min_intersections = min_intersections
        self.max_intersections = max_intersections
        self.only_complete = only_complete
        super().__init__()

    def _filter_scenario(self, scenario: Scenario) -> bool:
        # enforce minimum requirements for a scenario

        remove_empty_intersections(scenario.lanelet_network)
        num_intersections = len(scenario.lanelet_network.intersections)

        if self.only_complete:
            for intersection in scenario.lanelet_network.intersections:
                successors = get_intersection_successors(intersection)
                for lanelet_id in successors:
                    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                    if not lanelet.successor or not lanelet.predecessor:
                        return False
                
                # for incoming_element in intersection.incomings:
                #     if len(incoming_element.incoming_lanelets) == 0:
                #         return False
                #     if len(
                #         set.union(
                #             incoming_element.successors_left,
                #             incoming_element.successors_right,
                #             incoming_element.successors_straight
                #         )
                #     ) == 0:
                #         return False
                
        if (self.min_intersections is None or num_intersections >= self.min_intersections) and \
           (self.max_intersections is None or num_intersections <= self.max_intersections):
            return True
        return False
