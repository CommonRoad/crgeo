from __future__ import annotations

from typing import Optional, Tuple, Type, TYPE_CHECKING

import numpy as np
import networkx as nx
from commonroad.scenario.lanelet import LaneletNetwork

from crgeo.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph, CanonicalTransform, canonical_graph_transform
from crgeo.dataset.extraction.road_network.implementations.intersection_graph.graph_conversion import create_intersection_graph_from_lanelet_network
from crgeo.common.geometry.helpers import relative_orientation


class IntersectionGraph(BaseRoadNetworkGraph):

    @classmethod
    def from_lanelet_network(
        cls,
        lanelet_network: LaneletNetwork,
        include_diagonals: bool = False,
        include_adjacent: bool = False,
        validate: bool = False,
        decimal_precision: int = 0,
        adjacency_cardinality_threshold: int = -1,
        lanelet_length_threshold: float = 0.0,
        initial_cleanup: bool = True
    ) -> BaseRoadNetworkGraph:
        graph, lanelet_network = create_intersection_graph_from_lanelet_network(
            lanelet_network=lanelet_network,
            include_diagonals=include_diagonals,
            include_adjacent=include_adjacent,
            validate=validate,
            decimal_precision=decimal_precision,
            adjacency_cardinality_threshold=adjacency_cardinality_threshold,
            lanelet_length_threshold=lanelet_length_threshold,
            initial_cleanup=initial_cleanup
        )
        intersection_graph = cls(graph=graph, lanelet_network=lanelet_network)
        return intersection_graph

    def get_canonical_graph(
        self,
        source: Tuple[int, int],
        include_radius: Optional[float] = 100.0,
        depth: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        distance_offset: float = 0.0,
        transform_mode: Optional[CanonicalTransform] = CanonicalTransform.TranslateRotateRescale
    ) -> Tuple[Type[BaseRoadNetworkGraph], LaneletNetwork]:

        # source_lanelet = self._original_lanelet_network.find_lanelet_by_id(self._edge_mapping[source])
        start_pos = self._node_attr[source[0]]['node_position']
        end_pos = self._node_attr[source[1]]['node_position']

        # if include_radius is None:
        canonical_graph = self._get_canonical_nx_graph(
            source=source,
            depth=depth,
            min_size=min_size,
            max_size=max_size,
            radius=include_radius
        )
        intermediate_lanelet_network = self.lanelet_network

        if transform_mode is not None:
            pos = np.array(list(nx.get_node_attributes(canonical_graph, 'node_position').values()))
            pos, rotation = canonical_graph_transform(
                pos=pos,
                mode=transform_mode,
                origin=start_pos,
                end=end_pos,
                distance_offset=distance_offset
            )
            pos_attr = {n: pos[i] for i, n in enumerate(canonical_graph.nodes())}
            nx.set_node_attributes(canonical_graph, pos_attr, name='node_position')

            exit_angles = np.array(list(nx.get_edge_attributes(canonical_graph, name='exit_angle').values()))
            if exit_angles:
                exit_angles_rotated = relative_orientation(exit_angles, -rotation)
                exit_angles_attr = {e: exit_angles_rotated[i] for i, e in enumerate(canonical_graph.edges())}
                nx.set_edge_attributes(canonical_graph, exit_angles_attr, name='exit_angle')

            start_angles = np.array(list(nx.get_edge_attributes(canonical_graph, name='start_angle').values()))
            if start_angles:
                start_angles_rotated = relative_orientation(start_angles, -rotation)
                start_angles_attr = {e: start_angles_rotated[i] for i, e in enumerate(canonical_graph.edges())}
                nx.set_edge_attributes(canonical_graph, start_angles_attr, name='start_angle')

        canonical_graph = self.__class__(canonical_graph)
        canonical_graph.lanelet_network = self.lanelet_network
        canonical_graph._scenario = self.scenario
        return canonical_graph, intermediate_lanelet_network
