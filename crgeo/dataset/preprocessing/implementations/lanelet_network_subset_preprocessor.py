from __future__ import annotations
from collections import defaultdict

from typing import Callable, Optional, Set, Tuple, Union, List
from random import choice

from commonroad.scenario.scenario import Scenario
from crgeo.common.geometry.helpers import resample_polyline
from crgeo.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet
from crgeo.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from numpy.linalg import norm
from crgeo.common.io_extensions.lanelet_network import (
    map_out_lanelets_to_intersections,
    map_successor_lanelets_to_intersections
)
from typing import TYPE_CHECKING

import numpy as np

 # TODO: Support connectivity-based filtering instead of radius-based
class LaneletNetworkSubsetPreprocessor(BaseScenarioPreprocessor):
    """
    Randomly selects a subsets of the lanelet network and discards the rest.
    """

    def __init__(
        self, 
        radius: Optional[float] = 75.0,
        max_hops: Optional[int] = None,
        center_intersections: bool = True,
        radius_cutoff_multiplier: float = 1.5,
        min_leaf_lanelet_length: float = 20.0
    ) -> None:
        self.radius = radius if radius is not None else np.inf
        self.radius_cutoff = self.radius * radius_cutoff_multiplier
        self.max_hops = max_hops
        self.center_intersections = center_intersections
        self.radius_cutoff_multiplier = radius_cutoff_multiplier
        self.min_leaf_lanelet_length = min_leaf_lanelet_length
        super(LaneletNetworkSubsetPreprocessor, self).__init__()

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        
        origin_coordinates = None

        if self.center_intersections and scenario.lanelet_network.intersections:
            origin_intersection = choice(scenario.lanelet_network.intersections)
            # finding coordinates
            for incoming_element in origin_intersection.incomings:
                if incoming_element.incoming_lanelets:
                    origin_lanelet = scenario.lanelet_network.find_lanelet_by_id(
                       list(incoming_element.incoming_lanelets)[0]
                    )
                    origin_coordinates = origin_lanelet.center_vertices[-1]
                    break
        if origin_coordinates is None:
            origin_lanelet = choice(scenario.lanelet_network.lanelets)
            origin_coordinates = origin_lanelet.center_vertices[0]

        if self.max_hops is not None:
            lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)
            proximities = lanelet_graph._proximity_matrix[origin_lanelet.lanelet_id]

        lanelet_ids_to_remove: Set[int] = set()
        lanelet_ids_to_remove_force: Set[int] = set()
        for lanelet in scenario.lanelet_network.lanelets:

            # radius check
            within_circle = norm(lanelet.center_vertices - origin_coordinates, axis=1) < self.radius
            if within_circle.sum() < 3:
                lanelet_ids_to_remove.add(lanelet.lanelet_id)
            
            index_filter = norm(lanelet.center_vertices - origin_coordinates, axis=1) < self.radius_cutoff
            if index_filter.sum() < 3:
                lanelet_ids_to_remove_force.add(lanelet.lanelet_id)
            else:
                intervals = max(30, lanelet.center_vertices.shape[0])
                lanelet._center_vertices = np.array(
                    resample_polyline(lanelet.center_vertices[index_filter], intervals)
                )
                lanelet._left_vertices = np.array(
                    resample_polyline(lanelet.left_vertices[index_filter], intervals)
                )
                lanelet._right_vertices = np.array(
                    resample_polyline(lanelet.right_vertices[index_filter], intervals)
                )

                # hack for recomputing
                lanelet._distance = None
                lanelet._inner_distance = None
                lanelet.distance
                lanelet.inner_distance

            # connectivity check
            if self.max_hops is not None:
                if lanelet.lanelet_id not in proximities or proximities[lanelet.lanelet_id] >= self.max_hops:
                    lanelet_ids_to_remove.add(lanelet.lanelet_id)

        out_lanelet_map = map_out_lanelets_to_intersections(scenario.lanelet_network)
        successor_lanelet_map = map_successor_lanelets_to_intersections(scenario.lanelet_network, validate=False)
        intersection_lanelet_map = defaultdict(set)
        for lanelet_id, intersection in out_lanelet_map.items():
            intersection_lanelet_map[intersection.intersection_id].add(lanelet_id)
        for lanelet_id, intersection in successor_lanelet_map.items():
            intersection_lanelet_map[intersection.intersection_id].add(lanelet_id)

        for intersection in scenario.lanelet_network.intersections:
            keep = False
            for incoming_element in intersection.incomings:
                for incoming_lanelet_id in incoming_element.incoming_lanelets:
                    if incoming_lanelet_id not in lanelet_ids_to_remove:
                        keep = True
                        break
            if not keep:
                scenario.lanelet_network.remove_intersection(intersection.intersection_id)
            else:
                for lanelet_id in intersection.map_incoming_lanelets:
                    if lanelet_id not in lanelet_ids_to_remove_force:
                        lanelet_ids_to_remove.discard(lanelet_id)
                for lanelet_id in intersection_lanelet_map[intersection.intersection_id]:
                    if lanelet_id not in lanelet_ids_to_remove_force:
                        lanelet_ids_to_remove.discard(lanelet_id)

        for lanelet_id in lanelet_ids_to_remove:
            scenario.lanelet_network.remove_lanelet(lanelet_id=lanelet_id)
        scenario.lanelet_network.cleanup_lanelet_references()

        for lanelet in scenario.lanelet_network.lanelets:
            if (not lanelet.predecessor or not lanelet.successor) and lanelet.distance[-1] < self.min_leaf_lanelet_length:
                scenario.lanelet_network.remove_lanelet(lanelet_id=lanelet.lanelet_id)

        scenario.lanelet_network.cleanup_lanelet_references()
        return scenario, planning_problem_set
