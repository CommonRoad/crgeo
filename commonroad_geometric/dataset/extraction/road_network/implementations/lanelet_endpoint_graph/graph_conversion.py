from typing import Callable, Dict, Set, Tuple, Union

import networkx as nx
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from shapely.geometry import LineString, MultiLineString, MultiPoint

from commonroad_geometric.common.geometry import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import chaikins_corner_cutting, cut_polyline, resample_polyline
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletNodeType


class LaneletEndpointGraphConverter:
    def __init__(
        self,
        conversion_steps: Tuple[Callable] = (),
        decimal_precision: int = 2,
        waypoint_density: Union[float, int] = 50,
        max_iterations: int = 1e6
    ) -> None:
        """

        Args:
            conversion_steps (Tuple[Callable]): Conversion functions  references operating on the graph in the order they are supposed to be called
            decimal_precision (int): Precision of coordinates for inserted conflict nodes.
            waypoint_density (Union[float, int]): Waypoint density of the lanelet polyline.
            max_iterations (int): Maximum allowed iterations.
        """
        self.conversion_steps = conversion_steps
        self.decimal_precision = decimal_precision
        self.waypoint_density = waypoint_density
        self.max_iterations = max_iterations
        self._position_to_node: Dict[Tuple[float, float], int] = {}
        self._lanelet_id_to_node: Dict[int, int] = {}

    def _insert_conflict_nodes(
        self,
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork  # unused but most pragmatic solution to be consistent with other functions
    ) -> nx.DiGraph:
        """Inserts conflict nodes where lanelet edges are intersecting.

        Args:
            graph (nx.DiGraph): Lanelet graph.
            lanelet_network: LaneletNetwork

        Raises:
            RuntimeError: If procedure does not halt before max_iterations.

        Returns:
            nx.DiGraph: New graph instance with conflict nodes.
        """
        extended_graph = graph.copy()

        inserted_positions: Set[Tuple[float, float]] = set()

        def find_and_insert_next_conflict_node(
            g: nx.DiGraph,
            iteration: int,
            checked_edge_pairs: Set[Tuple[int, int]]
        ) -> Tuple[bool, Set[Tuple[int, int]]]:
            nonlocal inserted_positions
            edge_lanelet_ids = nx.get_edge_attributes(g, 'lanelet_id')
            edge_waypoints = nx.get_edge_attributes(g, 'edge_waypoints')
            current_edges = list(g.edges)
            edge_to_linestring = {edge: LineString(edge_waypoints[edge]) for edge in current_edges}

            # Iterating over all edges
            for edge in current_edges:
                edge_shape = edge_to_linestring[edge]

                # Checking against all other edges
                for other_edge in current_edges:
                    if (edge, other_edge) in checked_edge_pairs:
                        continue
                    checked_edge_pairs.add((edge, other_edge))
                    if edge == other_edge:
                        continue
                    other_edge_shape = edge_to_linestring[other_edge]
                    intersection = edge_shape.intersection(other_edge_shape)
                    if isinstance(intersection, MultiPoint):
                        intersection = intersection[0]
                    if not intersection.is_empty and not isinstance(intersection, (LineString, MultiLineString)):
                        edge_projection = edge_shape.project(intersection)
                        other_edge_projection = other_edge_shape.project(intersection)
                        if edge_projection <= 0:
                            continue
                        if edge_projection >= edge_shape.length:
                            continue
                        if other_edge_projection <= 0:
                            continue
                        if other_edge_projection >= other_edge_shape.length:
                            continue

                        # Found intersection point corresponding to conflicting lanelets
                        conflict_node_ids = [
                            f"{iteration}-C-1",
                            f"{iteration}-C-2"
                        ]
                        conflict_node_pos = np.array([intersection.x, intersection.y]).round(self.decimal_precision)
                        conflict_node_pos_key = tuple(conflict_node_pos.tolist())
                        if conflict_node_pos_key in inserted_positions:
                            continue
                        inserted_positions.add(conflict_node_pos_key)

                        # Adding conflict nodes
                        for id in conflict_node_ids:
                            g.add_node(
                                id,
                                node_position=conflict_node_pos,
                                node_type=LaneletNodeType.CONFLICTING
                            )

                        # Adding links between conflict nodes
                        g.add_edge(
                            conflict_node_ids[0],
                            conflict_node_ids[1],
                            lanelet_edge_type=LaneletEdgeType.CONFLICT_LINK,
                            weight=0.0,
                            lanelet_id=-1,
                            edge_waypoints=None
                        )
                        g.add_edge(
                            conflict_node_ids[1],
                            conflict_node_ids[0],
                            lanelet_edge_type=LaneletEdgeType.CONFLICT_LINK,
                            weight=0.0,
                            lanelet_id=-1,
                            edge_waypoints=None
                        )

                        for e, conflict_node_id, shape, projection in zip(
                            [edge, other_edge],
                            conflict_node_ids,
                            [edge_shape, other_edge_shape],
                            [edge_projection, other_edge_projection]
                        ):
                            cut_polylines = cut_polyline(shape, projection)
                            source_polyline = ContinuousPolyline(cut_polylines[0], waypoint_resolution=self.waypoint_density)
                            target_polyline = ContinuousPolyline(cut_polylines[1], waypoint_resolution=self.waypoint_density)
                            # Adding edge from source node to conflict node with half weight of original edge
                            g.add_edge(
                                e[0],
                                conflict_node_id,
                                lanelet_edge_type=LaneletEdgeType.CONFLICTING,
                                weight=source_polyline.length,
                                lanelet_id=edge_lanelet_ids[e],
                                edge_waypoints=source_polyline.init_waypoints
                            )

                            # Adding edge from conflict node to target node with other half of weight of original edge
                            g.add_edge(
                                conflict_node_id,
                                e[1],
                                lanelet_edge_type=LaneletEdgeType.CONFLICTING,
                                weight=target_polyline.length,
                                lanelet_id=edge_lanelet_ids[e],
                                edge_waypoints=target_polyline.init_waypoints
                            )

                            # Removing original edge
                            g.remove_edge(
                                e[0],
                                e[1]
                            )

                        return True, checked_edge_pairs
            return False, checked_edge_pairs

        outstanding_fixes = True
        iteration = 0
        checked_edge_pairs = set()
        while outstanding_fixes:
            outstanding_fixes, checked_edge_pairs = find_and_insert_next_conflict_node(
                extended_graph,
                iteration=iteration,
                checked_edge_pairs=checked_edge_pairs
            )
            iteration += 1
            if iteration > self.max_iterations:
                raise RuntimeError("Conflict nodes insertion procedure likely stuck")

        return extended_graph

    def _connect_adjacent_nodes(
            self,
            graph: nx.DiGraph,
            lanelet_network: LaneletNetwork
    ) -> nx.DiGraph:
        """
        Add lateral bi-directed edges between the adjacent nodes in the graph

        Args:
            graph: (nx.DiGraph)
            lanelet_network: (LaneletNetwork)

        Returns:
            graph (nx.DiGraph) - the graph with connections between adjacent nodes
        """
        lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)

        for lanelet in lanelet_network.lanelets:
            if lanelet.adj_left_same_direction:
                if lanelet.adj_left not in lanelet_ids:
                    continue
                adj_left = lanelet_network.find_lanelet_by_id(lanelet.adj_left)

                lanelet_width = np.linalg.norm(lanelet.left_vertices[-1] - lanelet.right_vertices[-1])
                adj_left_width = np.linalg.norm(adj_left.left_vertices[-1] - adj_left.right_vertices[-1])
                adj_distance = np.mean([lanelet_width, adj_left_width])

                graph.add_edge(
                    self._lanelet_id_to_node[lanelet.lanelet_id],
                    self._lanelet_id_to_node[lanelet.adj_left],
                    lanelet_edge_type=LaneletEdgeType.ADJACENT_LEFT,
                    weight=adj_distance,
                    lanelet_id=-1,
                    lanelets=[],
                    edge_waypoints=np.vstack([
                        lanelet.center_vertices[0],
                        adj_left.center_vertices[0]
                    ])
                )
            if lanelet.adj_right_same_direction:
                if lanelet.adj_right not in lanelet_ids:
                    continue
                adj_right = lanelet_network.find_lanelet_by_id(lanelet.adj_right)

                lanelet_width = np.linalg.norm(lanelet.left_vertices[-1] - lanelet.right_vertices[-1])
                adj_right_width = np.linalg.norm(adj_right.left_vertices[-1] - adj_right.right_vertices[-1])
                adj_distance = np.mean([lanelet_width, adj_right_width])

                graph.add_edge(
                    self._lanelet_id_to_node[lanelet.lanelet_id],
                    self._lanelet_id_to_node[lanelet.adj_right],
                    lanelet_edge_type=LaneletEdgeType.ADJACENT_RIGHT,
                    weight=adj_distance,
                    lanelet_id=-1,
                    lanelets=[],
                    edge_waypoints=np.vstack([
                        lanelet.center_vertices[0],
                        adj_right.center_vertices[0]
                    ])
                )
        return graph

    def _add_diagonal_edges(
        self,
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork
    ) -> nx.DiGraph:
        """
        Add diagonal edges between the start of a lanelet and the end of its adjacent lanelets in the graph.

        Args:
            graph: (nx.DiGraph)
            lanelet_network: (LaneletNetwork)

        Returns:
            graph (nx.DiGraph) - the graph with diagonal edges between adjacent nodes
        """
        lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)
        node_to_position = nx.get_node_attributes(graph, 'node_position')

        for lanelet in lanelet_network.lanelets:
            # Diagonal edge between lanelet and successor of adj_left
            if lanelet.adj_left_same_direction:
                if lanelet.adj_left not in lanelet_ids:
                    continue
                adj_left = lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                if adj_left.successor:
                    for adj_left_successor_id in adj_left.successor:
                        if adj_left_successor_id not in lanelet_ids:
                            continue
                        adj_left_successor = lanelet_network.find_lanelet_by_id(adj_left_successor_id)

                        lanelet_node = self._lanelet_id_to_node[lanelet.lanelet_id]
                        adj_left_node = self._lanelet_id_to_node[adj_left_successor_id]

                        lanelet_position = np.array(node_to_position[lanelet_node])
                        adj_left_position = np.array(node_to_position[adj_left_node])
                        weight = np.linalg.norm(lanelet_position - adj_left_position)

                        graph.add_edge(
                            lanelet_node,
                            adj_left_node,
                            lanelet_edge_type=LaneletEdgeType.DIAGONAL_LEFT,
                            weight=weight,
                            lanelet_id=adj_left_successor_id,
                            lanelets=[],
                            edge_waypoints=np.vstack([
                                lanelet.center_vertices[0],
                                adj_left_successor.center_vertices[0]
                            ])
                        )
            if lanelet.adj_right_same_direction:
                if lanelet.adj_right not in lanelet_ids:
                    continue
                adj_right = lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                # Diagonal edge between lanelet and successor of adj_right
                if adj_right.successor:
                    for adj_right_successor_id in adj_right.successor:
                        if adj_right_successor_id not in lanelet_ids:
                            continue
                        adj_right_successor = lanelet_network.find_lanelet_by_id(adj_right_successor_id)

                        lanelet_node = self._lanelet_id_to_node[lanelet.lanelet_id]
                        adj_right_node = self._lanelet_id_to_node[adj_right_successor_id]

                        lanelet_position = np.array(node_to_position[lanelet_node])
                        adj_left_position = np.array(node_to_position[self._lanelet_id_to_node[adj_right_node]])
                        weight = np.linalg.norm(lanelet_position - adj_left_position)

                        graph.add_edge(
                            lanelet_node,
                            adj_right_node,
                            lanelet_edge_type=LaneletEdgeType.DIAGONAL_RIGHT,
                            weight=weight,
                            lanelet_id=adj_right_successor_id,
                            lanelets=[],
                            edge_waypoints=np.vstack([
                                lanelet.center_vertices[0],
                                adj_right_successor.center_vertices[0]
                            ])
                        )
        return graph

    def create_lanelet_endpoint_graph_from_lanelet_network(
        self,
        lanelet_network: LaneletNetwork
    ) -> Tuple[nx.DiGraph, LaneletNetwork]:
        # Reset from previous call
        self._position_to_node = {}
        self._lanelet_id_to_node = {}
        graph = nx.DiGraph()

        # Inserting nodes for beginning of lanelets
        for lanelet in lanelet_network.lanelets:
            pos = tuple(lanelet.center_vertices[0].round(self.decimal_precision).tolist())
            if pos not in self._position_to_node:
                self._position_to_node[pos] = lanelet.lanelet_id
                graph.add_node(
                    lanelet.lanelet_id,
                    node_position=pos,
                    node_type=LaneletNodeType.LANELET_BEGINNING
                )
                self._lanelet_id_to_node[lanelet.lanelet_id] = lanelet.lanelet_id
            else:
                # Corresponding node has already been added
                self._lanelet_id_to_node[lanelet.lanelet_id] = self._position_to_node[pos]

        # Inserting nodes for ends of lanelets
        for lanelet in lanelet_network.lanelets:
            pos = tuple(lanelet.center_vertices[-1].round(self.decimal_precision).tolist())
            end_of_lanelet_id = f"{lanelet.lanelet_id}-E"
            if pos not in self._position_to_node:
                # Using "-E" suffix to represent lanelet ends
                self._position_to_node[pos] = end_of_lanelet_id
                graph.add_node(
                    end_of_lanelet_id,
                    node_position=pos,
                    node_type=LaneletNodeType.LANELET_END
                )
                self._lanelet_id_to_node[end_of_lanelet_id] = end_of_lanelet_id
            else:
                # Corresponding node has already been added
                self._lanelet_id_to_node[end_of_lanelet_id] = self._position_to_node[pos]

        # Adding edges
        for lanelet in lanelet_network.lanelets:
            lanelet_polyline = chaikins_corner_cutting(lanelet.center_vertices, 1)
            lanelet_polyline = np.array(resample_polyline(lanelet_polyline, self.waypoint_density))
            end_of_lanelet_id = f"{lanelet.lanelet_id}-E"
            # Successor edge
            graph.add_edge(
                self._lanelet_id_to_node[lanelet.lanelet_id],
                self._lanelet_id_to_node[end_of_lanelet_id],
                lanelet_edge_type=LaneletEdgeType.SUCCESSOR,
                weight=lanelet.distance[-1],
                lanelet_id=lanelet.lanelet_id,
                lanelets=[lanelet.lanelet_id],
                edge_waypoints=lanelet_polyline
            )

        for conversion_step in self.conversion_steps:
            graph = conversion_step(self, graph=graph, lanelet_network=lanelet_network)

        return graph, lanelet_network
