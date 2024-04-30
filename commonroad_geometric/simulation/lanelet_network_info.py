from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np
from collections import defaultdict
import networkx as nx
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.scenario import Scenario
from torch_geometric.data import Data

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.io_extensions.lanelet_network import find_lanelet_by_id
from commonroad_geometric.dataset.extraction.road_network.implementations import LaneletGraph
from commonroad_geometric.dataset.extraction.road_network.types import GraphConversionStep


class LaneletNetworkInfo(AutoReprMixin):
    """
    Helper class for managing the lanelet network in a simulation context.
    """
    # TODO this isn't so pretty

    def __init__(
        self,
        scenario: Scenario,
        graph_conversion_steps: Optional[List[GraphConversionStep]] = None
    ) -> None:

        self._scenario: Scenario = scenario
        self._lanelet_network: LaneletNetwork = scenario.lanelet_network
        self._lanelet_graph = LaneletGraph.from_lanelet_network(
            lanelet_network=scenario.lanelet_network,
            graph_conversion_steps=graph_conversion_steps,
            add_adj_opposite_dir=False
        )
        self._lanelet_hops = dict(nx.shortest_path_length(self._lanelet_graph))
        self._traffic_flow_graph = self._lanelet_graph.get_traffic_flow_graph()
        self._lanelet_graph_data: Data = self._lanelet_graph.get_torch_data()

        self._routes = self._extract_routes()

        self._lanelet_id_to_lanelet_idx: Dict[int, int] = {
            lanelet_id.item(): idx for idx, lanelet_id in enumerate(self._lanelet_graph_data["id"])
        }

        self._lanelet_center_polylines: Dict[int, ContinuousPolyline] = {}
        self._lanelet_left_polylines: Dict[int, ContinuousPolyline] = {}
        self._lanelet_right_polylines: Dict[int, ContinuousPolyline] = {}

    @property
    def lanelet_scenario(self) -> Scenario:
        return self._scenario

    @property
    def lanelet_network(self) -> LaneletNetwork:
        return self._lanelet_network

    @property
    def lanelet_graph(self) -> LaneletGraph:
        return self._lanelet_graph

    @property
    def traffic_flow_graph(self) -> LaneletGraph:
        return self._traffic_flow_graph

    @property
    def lanelet_graph_data(self) -> Data:
        return self._lanelet_graph_data

    @property
    def lanelet_id_to_lanelet_idx(self) -> Dict[int, int]:
        return self._lanelet_id_to_lanelet_idx

    @property
    def lanelet_hops(self) -> Dict[int, Dict[int, int]]:
        return self._lanelet_hops

    @property
    def num_lanelets(self) -> int:
        return len(self._scenario.lanelet_network.lanelets)

    @property
    def routes(self) -> Dict[int, Dict[int, List[int]]]:
        return self._routes

    @property
    def entry_lanelet_ids(self) -> Set[int]:
        return self._entry_lanelet_ids

    @property
    def exit_lanelet_ids(self) -> Set[int]:
        return self._exit_lanelet_ids

    @property
    def lanelets(self) -> List[Lanelet]:
        return self._scenario.lanelet_network.lanelets

    def _extract_routes(self) -> Dict[int, Dict[int, List[int]]]:
        self._entry_lanelet_ids = set([lanelet.lanelet_id for lanelet in self.lanelets if not lanelet.predecessor])
        self._exit_lanelet_ids = set([lanelet.lanelet_id for lanelet in self.lanelets if not lanelet.successor])
        shortest_path = nx.shortest_path(self._traffic_flow_graph)
        routes: Dict[int, Dict[int, List[int]]] = {}
        for from_lanelet_id in shortest_path:
            if from_lanelet_id not in self._entry_lanelet_ids:
                continue
            routes[from_lanelet_id] = {}
            for to_lanelet_id in shortest_path[from_lanelet_id]:
                if to_lanelet_id not in self._exit_lanelet_ids:
                    continue
                routes[from_lanelet_id][to_lanelet_id] = shortest_path[from_lanelet_id][to_lanelet_id]
        return routes
    
    def get_all_connected_lanelets(self) -> dict:
        """
        Create all possible lanes by merging predecessors and successors, then create a dict with its keys as lanelet id
        and values as connected lanelet ids.
        
        :return: dict
        """
        merged_lanelet_dict = defaultdict(set)
        for l in self.lanelet_network.lanelets:  # iterate in all lanelet in this scenario
            if not l.predecessor and not l.successor:  # the lanelet is a lane itself
                merged_lanelet_dict[l.lanelet_id].add(l.lanelet_id)
            elif not l.predecessor:
                max_lane_merge_range = 1000.0
                _, sub_lanelet_ids = Lanelet.all_lanelets_by_merging_successors_from_lanelet(l, self.lanelet_network,
                                                                                            max_lane_merge_range)
                for s in sub_lanelet_ids:
                    for i in s:
                        merged_lanelet_dict[i].update(s)
        return merged_lanelet_dict

    def find_lanelet_by_id(self, lanelet_id: int) -> Lanelet:
        return find_lanelet_by_id(self._lanelet_network, lanelet_id)

    def get_lanelet_center_polyline(self, lanelet_id: int, vertices: Optional[np.ndarray] = None) -> ContinuousPolyline:
        if lanelet_id not in self._lanelet_center_polylines:
            vertices = vertices if vertices is not None else self.find_lanelet_by_id(lanelet_id).center_vertices
            self._lanelet_center_polylines[lanelet_id] = ContinuousPolyline(vertices)
        return self._lanelet_center_polylines[lanelet_id]

    def get_lanelet_left_polyline(self, lanelet_id: int, vertices: Optional[np.ndarray] = None) -> ContinuousPolyline:
        if lanelet_id not in self._lanelet_left_polylines:
            vertices = vertices if vertices is not None else self.find_lanelet_by_id(lanelet_id).left_vertices
            self._lanelet_left_polylines[lanelet_id] = ContinuousPolyline(vertices)
        return self._lanelet_left_polylines[lanelet_id]

    def get_lanelet_right_polyline(self, lanelet_id: int, vertices: Optional[np.ndarray] = None) -> ContinuousPolyline:
        if lanelet_id not in self._lanelet_right_polylines:
            vertices = vertices if vertices is not None else self.find_lanelet_by_id(lanelet_id).right_vertices
            self._lanelet_right_polylines[lanelet_id] = ContinuousPolyline(vertices)
        return self._lanelet_right_polylines[lanelet_id]
