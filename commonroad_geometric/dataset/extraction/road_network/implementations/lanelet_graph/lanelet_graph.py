from __future__ import annotations

import functools
from typing import Iterable, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph, CanonicalTransform, canonical_graph_transform
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.graph_conversion import LaneletGraphConverter
from commonroad_geometric.dataset.extraction.road_network.types import GraphConversionStep


class LaneletGraph(BaseRoadNetworkGraph):
    @classmethod
    def from_lanelet_network(
        cls,
        lanelet_network: LaneletNetwork,
        graph_conversion_steps: Optional[Iterable[GraphConversionStep]] = None,
        waypoint_density: Union[float, int] = 50,
        add_adj_opposite_dir: bool = False
    ) -> BaseRoadNetworkGraph:
        if graph_conversion_steps is None:
            graph_conversion_steps = (
                functools.partial(LaneletGraphConverter.resample_waypoints, waypoint_density=waypoint_density),
                LaneletGraphConverter.connect_predecessor_successors,
                LaneletGraphConverter.connect_successor_predecessor,
                functools.partial(LaneletGraphConverter.insert_conflict_edges),
            )

        graph_converter = LaneletGraphConverter(
            graph_conversion_steps=graph_conversion_steps,
        )

        graph, lanelet_network = graph_converter.create_lanelet_graph_from_lanelet_network(
            lanelet_network=lanelet_network,
            add_adj_opposite_dir=add_adj_opposite_dir
        )
        lanelet_graph = cls(graph, lanelet_network=lanelet_network)
        return lanelet_graph

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

        waypoint_start_pos = self._node_attr[source[0]]['node_waypoints'][0]
        waypoint_end_pos = self._node_attr[source[1]]['node_waypoints'][0]

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

            # TODO: transform other coordinates as well? left, center, right vertices?
            node_waypoints = list(nx.get_node_attributes(canonical_graph, name='node_waypoints').values())
            pos_list = []
            for waypoint in node_waypoints:
                pos, _ = canonical_graph_transform(
                    pos=waypoint,
                    mode=transform_mode,
                    origin=waypoint_start_pos,
                    end=waypoint_end_pos,
                    distance_offset=distance_offset
                )
                pos_list.append(pos)
            waypoint_attr = {e: pos_list[i] for i, e in enumerate(canonical_graph.nodes())}
            nx.set_node_attributes(canonical_graph, waypoint_attr, name='node_waypoints')

            exit_angles = np.array(list(nx.get_edge_attributes(canonical_graph, name='exit_angle').values()))
            if len(exit_angles) != 0:
                exit_angles_rotated = relative_orientation(exit_angles, -rotation)
                exit_angles_attr = {e: exit_angles_rotated[i] for i, e in enumerate(canonical_graph.edges())}
                nx.set_edge_attributes(canonical_graph, exit_angles_attr, name='exit_angle')

            start_angles = np.array(list(nx.get_edge_attributes(canonical_graph, name='start_angle').values()))
            if len(start_angles) != 0:
                start_angles_rotated = relative_orientation(start_angles, -rotation)
                start_angles_attr = {e: start_angles_rotated[i] for i, e in enumerate(canonical_graph.edges())}
                nx.set_edge_attributes(canonical_graph, start_angles_attr, name='start_angle')

        canonical_graph = self.__class__(canonical_graph)
        canonical_graph.lanelet_network = self.lanelet_network
        canonical_graph._scenario = self.scenario
        return canonical_graph, intermediate_lanelet_network
