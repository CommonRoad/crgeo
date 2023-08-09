from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Set, TYPE_CHECKING, Tuple, Type, Union

import networkx as nx
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint

from commonroad_geometric.common.geometry.helpers import TWO_PI, chaikins_corner_cutting, relative_orientation, resample_polyline, rotate_2d_matrix
from commonroad_geometric.dataset.extraction.road_network.types import GraphConversionStep, LaneletEdgeType, LaneletNodeType

if TYPE_CHECKING:
    from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.road_renderer import RoadRenderer


class LaneletGraphConverter:

    def __init__(
        self,
        graph_conversion_steps: Iterable[GraphConversionStep] = (),
    ) -> None:
        """

        Args:
            graph_conversion_steps (Iterable[GraphConversionStep]): Conversion functions, operating on the graph in the order they are supposed to be called.
                                                An empty iterable represents no further modifications after the graph has been created.
        """
        self.graph_conversion_steps = graph_conversion_steps

    def create_lanelet_graph_from_lanelet_network(
        self,
        lanelet_network: LaneletNetwork,
        add_adj_opposite_dir: bool = True
    ) -> Tuple[nx.DiGraph, LaneletNetwork]:
        all_lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)
        graph = nx.DiGraph()

        for lanelet in lanelet_network.lanelets:
           

            assert lanelet.lanelet_id is not None

            start_pos = lanelet.center_vertices[0]
            center_pos, right_pos, left_pos, segment_id = lanelet.interpolate_position(lanelet.distance[-1] / 2)
            end_pos = lanelet.center_vertices[-1]

            lanelet_dir = lanelet.center_vertices[1] - lanelet.center_vertices[0]
            orientation = np.arctan2(lanelet_dir[1], lanelet_dir[0])

            graph.add_node(
                lanelet.lanelet_id,
                node_position=start_pos,
                node_type=LaneletNodeType.LANELET_BEGINNING,
                start_pos=start_pos,
                center_pos=center_pos,
                right_pos=right_pos,
                left_pos=left_pos,
                end_pos=end_pos,
                orientation=orientation,
                length=lanelet.distance[-1],
                lanelet_id=lanelet.lanelet_id,
                center_vertices=lanelet.center_vertices,
                left_vertices=lanelet.left_vertices,
                right_vertices=lanelet.right_vertices,
            )

            for predecessor_id in lanelet.predecessor:
                if predecessor_id not in all_lanelet_ids:
                    continue
                predecessor = lanelet_network.find_lanelet_by_id(predecessor_id)
                graph.add_edge(
                    lanelet.lanelet_id,
                    predecessor_id,
                    lanelet_edge_type=LaneletEdgeType.PREDECESSOR.value,
                    weight=predecessor.distance[-1],
                    lanelets=[lanelet.lanelet_id],
                    source_arclength=0.0,
                    target_arclength=predecessor.distance[-1],
                    source_arclength_rel=0.0,
                    target_arclength_rel=1.0,
                    edge_position=predecessor.center_vertices[-1]
                )
            for successor in lanelet.successor:
                if successor not in all_lanelet_ids:
                    continue
                graph.add_edge(
                    lanelet.lanelet_id,
                    successor,
                    lanelet_edge_type=LaneletEdgeType.SUCCESSOR.value,
                    weight=lanelet.distance[-1],
                    lanelets=[lanelet.lanelet_id],
                    source_arclength=lanelet.distance[-1],
                    target_arclength=0.0,
                    source_arclength_rel=1.0,
                    target_arclength_rel=0.0,
                    edge_position=lanelet.center_vertices[-1]
                )
            if lanelet.adj_left is not None:
                assert lanelet.adj_left in all_lanelet_ids
                lanelet_edge_type = LaneletEdgeType.ADJACENT_LEFT.value if lanelet.adj_left_same_direction \
                    else LaneletEdgeType.OPPOSITE_LEFT.value
                if lanelet.adj_left_same_direction or add_adj_opposite_dir:
                    graph.add_edge(
                        lanelet.lanelet_id,
                        lanelet.adj_left,
                        lanelet_edge_type=lanelet_edge_type,
                        weight=0.0,
                        lanelets=[],
                        source_arclength=0.0,
                        target_arclength=0.0,
                        source_arclength_rel=0.0,
                        target_arclength_rel=0.0,
                        edge_position=lanelet.left_vertices[0]
                    )
            if lanelet.adj_right:
                assert lanelet.adj_right in all_lanelet_ids
                lanelet_edge_type = LaneletEdgeType.ADJACENT_RIGHT.value if lanelet.adj_right_same_direction \
                    else LaneletEdgeType.OPPOSITE_RIGHT.value
                if lanelet.adj_right_same_direction or add_adj_opposite_dir:
                    graph.add_edge(
                        lanelet.lanelet_id,
                        lanelet.adj_right,
                        lanelet_edge_type=lanelet_edge_type,
                        weight=0.0,
                        lanelets=[],
                        source_arclength=0.0,
                        target_arclength=0.0,
                        source_arclength_rel=0.0,
                        target_arclength_rel=0.0,
                        edge_position=lanelet.right_vertices[0]
                    )

        for conversion_step in self.graph_conversion_steps:
            graph = conversion_step(graph=graph, lanelet_network=lanelet_network)

        return graph, lanelet_network

    @staticmethod
    def resample_waypoints(
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork,
        waypoint_density: Union[float, int],
        corner_cutting_iterations: int = 1,
    ) -> nx.DiGraph:
        for node in graph.nodes.values():
            for attr in ['left_vertices', 'center_vertices', 'right_vertices']:
                polyline = node[attr]
                if corner_cutting_iterations > 0:
                    polyline = chaikins_corner_cutting(polyline, corner_cutting_iterations)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    node[attr] = np.array(resample_polyline(polyline, waypoint_density))
        return graph

    @staticmethod
    def compute_lanelet_width(
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork,
    ) -> nx.DiGraph:
        for node in graph.nodes.values():
            node["width"] = np.linalg.norm(node["left_vertices"] - node["right_vertices"], axis=-1)
        return graph

    @staticmethod
    def insert_conflict_edges(
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork,  # unused but most pragmatic solution to be consistent with other functions
        waypoint_node_attribute: str = "center_vertices",
    ) -> nx.DiGraph:
        """Inserts conflict edges where lanelet nodes are intersecting.

        Args:
            graph (nx.DiGraph): Lanelet graph.
            lanelet_network: (LaneletNetwork)

        Returns:
            graph (nx.DiGraph) -  New graph instance with conflict nodes.
        """
        g = graph.copy()

        node_waypoints = nx.get_node_attributes(g, waypoint_node_attribute)
        node_to_linestring = {node: LineString(node_waypoints[node]) for node in g.nodes()}

        # Iterating over all nodes
        for node in g.nodes():
            node_shape = node_to_linestring[node]

            # Checking against all other nodes
            for other_node in g.nodes():
                if node == other_node:
                    continue
                other_node_shape = node_to_linestring[other_node]
                intersection = node_shape.intersection(other_node_shape)
                if isinstance(intersection, MultiPoint):
                    intersection = intersection.geoms[0]
                    # fixes lib/python3.9/site-packages/shapely/geometry/base.py", line 1000, in __getitem__:
                    #  "Iteration over multi-part geometries is deprecated and will be removed in "
                    #  "Shapely 2.0. Use the `geoms` property to access the constituent parts of "
                    #  "a multi-part geometry."
                if not intersection.is_empty and not isinstance(intersection, (LineString, MultiLineString)):
                    node_projection = node_shape.project(intersection)
                    other_node_projection = other_node_shape.project(intersection)
                    if node_projection <= 0:
                        continue
                    if node_projection >= node_shape.length:
                        continue
                    if other_node_projection <= 0:
                        continue
                    if other_node_projection >= other_node_shape.length:
                        continue
                    
                    location = np.array(node_shape.interpolate(node_projection).coords)

                    # Adding links between conflict nodes
                    g.add_edge(
                        node,
                        other_node,
                        lanelet_edge_type=LaneletEdgeType.CONFLICTING.value,
                        source_arclength=node_projection,
                        target_arclength=other_node_projection,
                        weight=0.0,
                        source_arclength_rel=node_projection / node_shape.length,
                        target_arclength_rel=other_node_projection / other_node_shape.length,
                        edge_position=location
                    )
                    g.add_edge(
                        other_node,
                        node,
                        lanelet_edge_type=LaneletEdgeType.CONFLICTING.value,
                        source_arclength=other_node_projection,
                        target_arclength=node_projection,
                        weight=0.0,
                        source_arclength_rel=other_node_projection / other_node_shape.length,
                        target_arclength_rel=node_projection / node_shape.length,
                        edge_position=location
                    )
        return g

    @staticmethod
    def connect_predecessor_successors(
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork
    ) -> nx.DiGraph:
        """

        Args:
            graph (nx.DiGraph): Lanelet graph.
            lanelet_network: (LaneletNetwork)

        Returns:
            graph (nx.DiGraph) -  Graph with direct edges between predecessors and successors of a lanelet.
        """
        lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)
        for lanelet in lanelet_network.lanelets:
            for predecessor in lanelet.predecessor:
                predecessor_lanelet = lanelet_network.find_lanelet_by_id(predecessor)
                for predecessor_successor in predecessor_lanelet.successor:
                    if predecessor_successor not in lanelet_ids or predecessor_successor == lanelet.lanelet_id:
                        continue
                    # predecessor_successor_lanelet = lanelet_network.find_lanelet_by_id(predecessor_successor)
                    graph.add_edge(
                        lanelet.lanelet_id,
                        predecessor_successor,
                        lanelet_edge_type=LaneletEdgeType.DIVERGING.value,
                        lanelets=[],
                        source_arclength=0.0,
                        target_arclength=0.0,
                        source_arclength_rel=0.0,
                        target_arclength_rel=0.0,
                        weight=0.0,
                        edge_position=lanelet.center_vertices[0]
                    )
        return graph

    @staticmethod
    def connect_successor_predecessor(
        graph: nx.DiGraph,
        lanelet_network: LaneletNetwork
    ) -> nx.DiGraph:
        """

        Args:
            graph (nx.DiGraph): Lanelet graph.
            lanelet_network: (LaneletNetwork)

        Returns:
            graph (nx.DiGraph) -  Graph with direct edges between predecessors and successors of a lanelet.
        """
        lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)
        for lanelet in lanelet_network.lanelets:
            for successor in lanelet.successor:
                successor_lanelet = lanelet_network.find_lanelet_by_id(successor)
                for successor_predecessor in successor_lanelet.predecessor:
                    if successor_predecessor not in lanelet_ids or successor_predecessor == lanelet.lanelet_id:
                        continue
                    successor_predecessor_lanelet = lanelet_network.find_lanelet_by_id(successor_predecessor)
                    graph.add_edge(
                        lanelet.lanelet_id,
                        successor_predecessor,
                        lanelet_edge_type=LaneletEdgeType.MERGING.value,
                        lanelets=[],
                        source_arclength=lanelet.distance[-1],
                        target_arclength=successor_predecessor_lanelet.distance[-1],
                        source_arclength_rel=1.0,
                        target_arclength_rel=1.0,
                        weight=0.0,
                        edge_position=lanelet.center_vertices[-1]
                    )
        return graph

    @staticmethod
    def lanelet_centered_coordinate_system(lanelet_network: LaneletNetwork, graph: nx.DiGraph) -> nx.DiGraph:
        for node in graph.nodes.values():
            lanelet_pos = node["start_pos"]
            # "start_pos", "center_pos", "end_pos", "orientation" are not relative to the lanelet reference frame

            rotation = rotate_2d_matrix(-node["orientation"])
            node["relative_center_vertices"] = (node["center_vertices"] - lanelet_pos) @ rotation.T
            node["relative_left_vertices"] = (node["left_vertices"] - lanelet_pos) @ rotation.T
            node["relative_right_vertices"] = (node["right_vertices"] - lanelet_pos) @ rotation.T

        for (u, v), edge in graph.edges.items():
            rel_orientation = relative_orientation(graph.nodes[u]["orientation"], graph.nodes[v]["orientation"])
            edge["relative_orientation"] = rel_orientation
            rotation = rotate_2d_matrix(-graph.nodes[u]["orientation"])
            relative_position = rotation @ (graph.nodes[v]["start_pos"] - graph.nodes[u]["start_pos"])
            edge["relative_position"] = relative_position
            edge["distance"] = np.linalg.norm(relative_position)

        return graph

    @staticmethod
    def lanelet_curvature(
        lanelet_network: LaneletNetwork,
        graph: nx.DiGraph,
        alpha: float = 0.6,
        max_depth: int = 2,
        lanelet_curvature_aggregation: Literal["sum", "abs", "sqr", "min", "max", "variance"] = "abs",
        curvature_aggregation: Literal["avg", "max", "min", "sum"] = "avg",
    ) -> nx.DiGraph:
        assert lanelet_curvature_aggregation in {"sum", "abs", "sqr", "min", "max", "variance"}
        assert curvature_aggregation in {"avg", "max", "min", "sum"}

        lanelet_curvature_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for node_id in graph.nodes:
            lanelet = lanelet_network.find_lanelet_by_id(node_id)
            center_vertices = lanelet.center_vertices
            curvatures, _ = compute_curvature(vertices=center_vertices)
            lanelet_curvature_data[node_id] = center_vertices, curvatures

        for node_id, node in graph.nodes.items():
            # depth-limited depth first computation with edge filter
            # based on nx.dfs_labeled_edges
            stack = [
                _LaneletCurvatureStackEntry(node_id, 0, iter(graph[node_id])),
            ]
            while stack:
                entry = stack[-1]
                if entry.depth == max_depth:
                    stack.pop()

                    # incorporate the curvature from the last segment of the preceding lanelet to
                    # the first segment of the current lanelet
                    center_vertices, curvatures = lanelet_curvature_data[entry.node_id]
                    preceding_center_vertices, _ = lanelet_curvature_data[stack[-1].node_id]
                    curvature = _compute_full_lanelet_curvature(
                        curr_curvatures=curvatures,
                        curr_vertices=center_vertices,
                        prec_vertices=preceding_center_vertices,
                        aggregation=lanelet_curvature_aggregation,
                    )
                    stack[-1].child_curvatures.append(curvature)
                    continue

                try:
                    child_id = next(entry.children)

                    # edge filter
                    if graph[entry.node_id][child_id]["lanelet_edge_type"] == LaneletEdgeType.SUCCESSOR:
                        stack.append(_LaneletCurvatureStackEntry(child_id, entry.depth + 1, iter(graph[child_id])))

                except StopIteration:
                    stack.pop()

                    center_vertices, curvatures = lanelet_curvature_data[entry.node_id]
                    if stack:
                        # incorporate the curvature from the last segment of the preceding lanelet to
                        # the first segment of the current lanelet
                        preceding_center_vertices, _ = lanelet_curvature_data[stack[-1].node_id]
                        lanelet_curvature = _compute_full_lanelet_curvature(
                            curr_curvatures=curvatures,
                            curr_vertices=center_vertices,
                            prec_vertices=preceding_center_vertices,
                            aggregation=lanelet_curvature_aggregation,
                        )
                    else:
                        lanelet_curvature = _aggregate_lanelet_curvature(
                            curvatures=curvatures,
                            aggregation=lanelet_curvature_aggregation,
                        )

                    if entry.child_curvatures:
                        if curvature_aggregation == "avg":
                            child_curvature = sum(entry.child_curvatures) / len(entry.child_curvatures)
                        elif curvature_aggregation == "max":
                            child_curvature = max(entry.child_curvatures)
                        elif curvature_aggregation == "min":
                            child_curvature = min(entry.child_curvatures)
                        elif curvature_aggregation == "sum":
                            child_curvature = sum(entry.child_curvatures)
                        lanelet_curvature = alpha * lanelet_curvature + (1 - alpha) * child_curvature

                    if stack:
                        stack[-1].child_curvatures.append(lanelet_curvature)
                    else:
                        node["curvature"] = lanelet_curvature

        return graph

    @staticmethod
    def render_road_coverage(
        lanelet_network: LaneletNetwork,
        graph: nx.DiGraph,
        size: int,
        lanelet_depth: int,
        lanelet_orientation_buckets: int = 0,
        edge_types: Optional[Set[LaneletEdgeType]] = None,
        renderer_cls: Optional[Type[RoadRenderer]] = None
    ) -> nx.DiGraph:
        if renderer_cls is None:
            from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.road_renderer import \
                RoadRenderer as renderer_cls

        if edge_types is None:
            edge_types = {
                LaneletEdgeType.SUCCESSOR,
                LaneletEdgeType.ADJACENT_LEFT,
                LaneletEdgeType.OPPOSITE_LEFT,
                LaneletEdgeType.ADJACENT_RIGHT,
                LaneletEdgeType.OPPOSITE_RIGHT,
                LaneletEdgeType.DIVERGING,
            }

        def edge_filter(e):
            return e["lanelet_edge_type"] in edge_types

        if lanelet_orientation_buckets > 0:
            node_segments_orientations = {}
            for node_id, node in graph.nodes.items():
                segment_directions = node["center_vertices"][1:] - node["center_vertices"][:-1]
                segment_orientations = np.arctan2(segment_directions[:, 1], segment_directions[:, 0])
                segment_orientations[segment_orientations < 0] += TWO_PI
                node_segments_orientations[node_id] = segment_orientations

        renderer = renderer_cls(size=size)
        for node_id, node in graph.nodes.items():
            render_node_ids = edge_filtered_bfs(
                graph,
                source_node=node_id,
                max_depth=lanelet_depth,
                edge_filter=edge_filter,
            )

            lanelet_position = node["start_pos"]
            lanelet_orientation = node["orientation"]

            if lanelet_orientation_buckets > 0:
                road_orientation = renderer.render_road_orientation(
                    graph=graph,
                    node_ids=render_node_ids,
                    scale=1.0,
                    lanelet_orientation=lanelet_orientation,
                    lanelet_position=lanelet_position,
                    lanelet_orientation_buckets=lanelet_orientation_buckets,
                    node_segments_orientations=node_segments_orientations,
                )
                node["road_orientation"] = road_orientation

            else:
                road_coverage = renderer.render_road_coverage(
                    graph=graph,
                    node_ids=render_node_ids,
                    scale=1.0,
                    lanelet_orientation=lanelet_orientation,
                    lanelet_position=lanelet_position,
                )
                node["road_coverage"] = road_coverage

        renderer.close()
        return graph


@dataclass
class _LaneletCurvatureStackEntry:
    node_id: Any
    depth: int
    children: Iterator
    child_curvatures: List[float] = field(default_factory=list)


def edge_filtered_bfs(
    graph: nx.DiGraph,
    source_node: int,
    max_depth: int,
    edge_filter: Callable[[dict], bool],
) -> Set[int]:
    assert max_depth >= 0
    current_nodes = {source_node}
    visited_nodes = {source_node}
    next_nodes: Set[int] = set()
    depth = 1
    while current_nodes and depth <= max_depth:
        for u in current_nodes:
            for v in graph[u]:
                if v in visited_nodes or not edge_filter(graph[u][v]):
                    continue

                next_nodes.add(v)

        visited_nodes.update(next_nodes)
        current_nodes, next_nodes = next_nodes, current_nodes
        next_nodes.clear()
        depth += 1

    return visited_nodes


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    # adapted from commonroad_geometric/external/commonroad_route_planner/utility/route.py
    if polyline.ndim != 2 or polyline.shape[0] <= 2:
        raise ValueError(f"Polyline malformed for curvature computation p={polyline}")

    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    # compute curvature
    tangent_vec_magnitude = x_d ** 2 + y_d ** 2
    tangent_vec_magnitude[np.isclose(tangent_vec_magnitude, 0.0)] = np.inf
    curvature = (x_d * y_dd - x_dd * y_d) / (tangent_vec_magnitude ** 1.5)
    return curvature


def compute_curvature(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    segments = vertices[1:] - vertices[:-1]
    if vertices.shape[0] <= 2:
        curvatures = np.zeros((vertices.shape[0],), dtype=vertices.dtype)
    else:
        curvatures = compute_curvature_from_polyline(vertices)
    return curvatures, segments


def _aggregate_lanelet_curvature(curvatures: np.ndarray, aggregation: str) -> float:
    if aggregation == "sum":
        return np.sum(curvatures).item()
    elif aggregation == "abs":
        return np.abs(curvatures).sum().item()
    elif aggregation == "sqr":
        return np.sum(curvatures ** 2).item()
    elif aggregation == "min":
        return np.min(curvatures).item()
    elif aggregation == "max":
        return np.max(curvatures).item()
    elif aggregation == "variance":
        return np.var(curvatures).item()
    raise ValueError(f"Unknown aggregation value {aggregation}")


def _compute_full_lanelet_curvature(
    curr_curvatures: np.ndarray,
    curr_vertices: np.ndarray,
    prec_vertices: np.ndarray,
    aggregation: str,
) -> float:
    # last vertex of preceding lanelet has to match the first vertex of the current lanelet
    assert np.allclose(prec_vertices[-1], curr_vertices[0])
    curvatures_from_preceding, _ = compute_curvature(
        vertices=np.array([prec_vertices[-2], curr_vertices[0], curr_vertices[1]]),
    )
    curvatures = np.concatenate([curvatures_from_preceding, curr_curvatures])
    return _aggregate_lanelet_curvature(curvatures, aggregation=aggregation)
