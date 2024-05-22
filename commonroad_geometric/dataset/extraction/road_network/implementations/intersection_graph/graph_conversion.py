from collections import defaultdict
from copy import deepcopy
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from numpy.linalg import norm

from commonroad_geometric.common.io_extensions.lanelet_network import (
    check_adjacent,
    check_connected,
    cleanup_intersections,
    map_lanelets_to_adjacency_clusters,
    map_out_lanelets_to_intersections,
    map_successor_lanelets_to_intersections,
    merge_successors,
    remove_empty_intersections
)
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletNodeType


class InvalidLaneletNetworkException(ValueError):
    pass


class UndesiredLaneletNetworkException(ValueError):
    pass


def create_intersection_graph_from_lanelet_network(
    lanelet_network: LaneletNetwork,
    include_diagonals: bool = False,
    include_adjacent: bool = False,
    validate: bool = False,
    decimal_precision: int = 0,
    adjacency_cardinality_threshold: int = -1,
    lanelet_length_threshold: float = 0.0,
    initial_cleanup: bool = True
) -> Tuple[nx.DiGraph, LaneletNetwork]:
    """Builds a graph from the lanelet network allowing diagonal lane changes.

    Args:
        lanelet_network (LaneletNetwork): Commonroad LaneletNetwork instance.
        include_adjacent (bool, optional): Draw edges between start nodes of adjacent edges. Defaults to False.
        validate (bool, optional): Whether to throw InvalidLaneletNetworkException if invalid lanelet network. Defaults to True.
        decimal_precision (int, optional): Decimal precision for node coordinates.
    Returns:
        nx.DiGraph: Greated graph from lanelet network with diagonal lane changes.
    """
    atol = 10 ** (-decimal_precision)
    graph = nx.DiGraph()
    node_pos: Dict[int, np.ndarray] = defaultdict(np.ndarray)
    node_lanelets: Dict[int, Set[int]] = defaultdict(set)
    node_mapping: Dict[int, int] = {}
    node_is_entry: Dict[int, int] = {}
    node_cardinality: Dict[int, int] = {}
    node_is_intersection: Dict[int, int] = {}
    overlap_mapping: Dict[Tuple[float, float], int] = {}
    edges = list()
    edge_attrs = list()
    edge_lanelets: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    lanelet_network = deepcopy(lanelet_network)
    lanelet_network.cleanup_lanelet_references()
    if initial_cleanup:
        remove_empty_intersections(lanelet_network)
        merge_successors(lanelet_network)
        cleanup_intersections(lanelet_network)
        merge_successors(lanelet_network)
        lanelet_network.cleanup_lanelet_references()
    if not lanelet_network.lanelets:
        raise UndesiredLaneletNetworkException("No lanelets left")
    min_distance = min([l.distance[-1] for l in lanelet_network.lanelets])
    if min_distance < lanelet_length_threshold:
        raise UndesiredLaneletNetworkException(f"Minimum lanelet length {min_distance:.2f} below threshold")

    intersection_ids = {intersection.intersection_id for intersection in lanelet_network.intersections}
    lanelet_ids = {lanelet.lanelet_id for lanelet in lanelet_network.lanelets}
    assert len(set.intersection(intersection_ids, lanelet_ids)) == 0

    intersection_incoming_map = lanelet_network.map_inc_lanelets_to_intersections
    intersection_outgoing_map = map_out_lanelets_to_intersections(lanelet_network)
    intersection_successor_map = map_successor_lanelets_to_intersections(lanelet_network)
    intersection_outgoing_map_only_left = map_out_lanelets_to_intersections(
        lanelet_network,
        include_right=False,
        include_straight=False
    )
    lanelet_adjacency_clusters, cluster_pos_map = map_lanelets_to_adjacency_clusters(
        lanelet_network,
        decimal_precision=decimal_precision
    )

    if len(lanelet_adjacency_clusters) > 0:
        max_adjacency_cardinality = max(len({w for w in v if w > 0}) for k, v in lanelet_adjacency_clusters.items())
    else:
        max_adjacency_cardinality = 1
    if adjacency_cardinality_threshold > -1 and max_adjacency_cardinality > adjacency_cardinality_threshold:
        raise UndesiredLaneletNetworkException(
            f"max_adjacency_cardinality {max_adjacency_cardinality} > {adjacency_cardinality_threshold}")
    intersection_outgoing_map_w_successors = {**intersection_outgoing_map, **intersection_successor_map}

    # Calculating intersection positions
    intersection_incoming_lanelet_ids = defaultdict(set)
    for lanelet_id, intersection in intersection_incoming_map.items():
        intersection_incoming_lanelet_ids[intersection.intersection_id].add(lanelet_id)
    intersection_all_outgoing_lanelet_ids = defaultdict(set)
    for lanelet_id, intersection in intersection_outgoing_map_w_successors.items():
        intersection_all_outgoing_lanelet_ids[intersection.intersection_id].add(lanelet_id)
    intersection_outgoing_lanelet_ids = defaultdict(set)
    for lanelet_id, intersection in intersection_outgoing_map.items():
        intersection_outgoing_lanelet_ids[intersection.intersection_id].add(lanelet_id)
    intersection_outgoing_left_lanelet_ids = defaultdict(set)
    for lanelet_id, intersection in intersection_outgoing_map_only_left.items():
        intersection_outgoing_left_lanelet_ids[intersection.intersection_id].add(lanelet_id)

    for intersection_id in intersection_all_outgoing_lanelet_ids:
        if intersection_id in intersection_outgoing_left_lanelet_ids == 0:
            intersection_lanelet_ids = intersection_outgoing_left_lanelet_ids[intersection_id]
        else:
            intersection_lanelet_ids = intersection_outgoing_lanelet_ids[intersection_id]
        for lanelet_id in intersection_lanelet_ids:
            node_mapping[lanelet_id] = intersection_id
        intersection_lanelets = {lanelet_network.find_lanelet_by_id(x) for x in intersection_lanelet_ids}
        intersection_coordinate_set = [l.interpolate_position(
            l.distance[-1] / 2)[0] for l in intersection_lanelets if l is not None]
        node_pos[intersection_id] = np.array(intersection_coordinate_set).mean(axis=0).round(decimal_precision)
        node_is_intersection[intersection_id] = True
        node_cardinality[intersection_id] = 1
        assert np.isfinite(node_pos[intersection_id]).all()

        for lanelet_id in intersection_all_outgoing_lanelet_ids[intersection_id]:
            node_mapping[lanelet_id] = intersection_id
        for lanelet_id in intersection_outgoing_lanelet_ids[intersection_id]:
            node_mapping[-lanelet_id] = intersection_id
        for lanelet_id in intersection_incoming_lanelet_ids[intersection_id]:
            node_mapping[-lanelet_id] = intersection_id

    # inserting missing intersections
    for intersection_id in intersection_incoming_lanelet_ids:
        if intersection_id not in node_pos:
            intersection_lanelet_ids = intersection_incoming_lanelet_ids[intersection_id]
            intersection_lanelets = {lanelet_network.find_lanelet_by_id(x) for x in intersection_lanelet_ids}
            intersection_coordinate_set = [l.interpolate_position(
                l.distance[-1] / 2)[0] for l in intersection_lanelets if l is not None]

            for lanelet_id in intersection_lanelet_ids:
                node_mapping[-lanelet_id] = intersection_id
            if len(intersection_coordinate_set):
                node_pos[intersection_id] = np.array(intersection_coordinate_set).mean(axis=0).round(decimal_precision)
                node_is_intersection[intersection_id] = True
                node_cardinality[intersection_id] = 1
                assert np.isfinite(node_pos[intersection_id]).all()

    utilized_clusters: Set[int] = set()
    for cluster_id, cluster_lanelet_ids in lanelet_adjacency_clusters.items():
        if cluster_id in node_mapping:
            continue

        pos = cluster_pos_map[cluster_id]
        overlap_tuple = tuple(pos.tolist())
        if overlap_tuple in overlap_mapping:
            mapped_cluster_id = overlap_mapping[overlap_tuple]
        else:
            overlap_mapping[overlap_tuple] = cluster_id
            mapped_cluster_id = cluster_id
            utilized_clusters.add(cluster_id)

        node_pos[cluster_id] = pos
        node_is_intersection[cluster_id] = False
        if cluster_id > 0:
            node_cardinality[cluster_id] = len([l for l in cluster_lanelet_ids if l > 0])
        else:
            node_cardinality[cluster_id] = len(cluster_lanelet_ids)
        for lanelet_id in cluster_lanelet_ids:
            node_mapping[lanelet_id] = mapped_cluster_id

    # inserting end nodes
    for lanelet in lanelet_network.lanelets:
        if -lanelet.lanelet_id in node_mapping:
            continue
        pos = lanelet.left_vertices[-1].round(decimal_precision)
        overlap_tuple = tuple(pos.tolist())
        if lanelet.lanelet_id in intersection_incoming_map:
            assert intersection_incoming_map[lanelet.lanelet_id].intersection_id in node_pos
            node_mapping[-lanelet.lanelet_id] = intersection_incoming_map[lanelet.lanelet_id].intersection_id
            overlap_mapping[overlap_tuple] = node_mapping[-lanelet.lanelet_id]
        if lanelet.lanelet_id in intersection_outgoing_map and lanelet.successor:
            assert intersection_outgoing_map[lanelet.lanelet_id].intersection_id in node_pos
            node_mapping[-lanelet.lanelet_id] = intersection_outgoing_map[lanelet.lanelet_id].intersection_id
            overlap_mapping[overlap_tuple] = node_mapping[-lanelet.lanelet_id]
        else:
            if overlap_tuple in overlap_mapping:
                node_mapping[-lanelet.lanelet_id] = overlap_mapping[overlap_tuple]
            else:
                overlap_mapping[overlap_tuple] = -lanelet.lanelet_id
                node_mapping[-lanelet.lanelet_id] = -lanelet.lanelet_id
                node_is_intersection[node_mapping[-lanelet.lanelet_id]] = False
                node_pos[node_mapping[-lanelet.lanelet_id]] = pos
                node_cardinality[node_mapping[-lanelet.lanelet_id]] = 1

    # inserting start nodes
    for lanelet in lanelet_network.lanelets:
        if lanelet.lanelet_id in node_mapping:
            continue
        if lanelet.adj_left and not lanelet.adj_left_same_direction:
            pos = lanelet.left_vertices[0].round(decimal_precision)
        elif len(lanelet.predecessor) == 1:
            predecessor = lanelet_network.find_lanelet_by_id(lanelet.predecessor[0])
            if predecessor.adj_left and not predecessor.adj_left_same_direction:
                pos = lanelet.left_vertices[0].round(decimal_precision)
        elif lanelet.adj_right and not lanelet.adj_right_same_direction:
            pos = lanelet.right_vertices[0].round(decimal_precision)
        else:
            pos = lanelet.center_vertices[0].round(decimal_precision)
        overlap_tuple = tuple(pos.tolist())

        if lanelet.lanelet_id in intersection_outgoing_map:
            assert intersection_all_outgoing_lanelet_ids[lanelet.lanelet_id].intersection_id in node_pos
            node_mapping[lanelet.lanelet_id] = intersection_outgoing_map[lanelet.lanelet_id].intersection_id
            overlap_mapping[overlap_tuple] = node_mapping[lanelet.lanelet_id]
        elif lanelet.lanelet_id in intersection_successor_map:
            assert intersection_all_outgoing_lanelet_ids[lanelet.lanelet_id].intersection_id in node_pos
            node_mapping[lanelet.lanelet_id] = intersection_successor_map[lanelet.lanelet_id].intersection_id
            overlap_mapping[overlap_tuple] = node_mapping[lanelet.lanelet_id]
        # elif len(lanelet.predecessor) == 1 and lanelet.predecessor[0] in intersection_outgoing_map:
        #     assert intersection_all_outgoing_lanelet_ids[lanelet.predecessor[0]].intersection_id in node_pos
        #     node_mapping[lanelet.lanelet_id] = intersection_outgoing_map[lanelet.predecessor[0]].intersection_id
        #     overlap_mapping[overlap_tuple] = node_mapping[lanelet.predecessor[0]]
        else:
            if overlap_tuple in overlap_mapping:
                node_mapping[lanelet.lanelet_id] = overlap_mapping[overlap_tuple]
            else:
                overlap_mapping[overlap_tuple] = lanelet.lanelet_id
                node_mapping[lanelet.lanelet_id] = lanelet.lanelet_id
                node_is_intersection[node_mapping[lanelet.lanelet_id]] = False
                node_pos[node_mapping[lanelet.lanelet_id]] = pos
                node_cardinality[node_mapping[lanelet.lanelet_id]] = 1

    for n in node_pos:
        node_is_entry[n] = 0

    for lanelet in lanelet_network.lanelets:
        lanelet_id = lanelet.lanelet_id
        n = node_mapping[lanelet.lanelet_id]
        node_lanelets[n].add(lanelet.lanelet_id)

        if not lanelet.predecessor:
            node_is_entry[n] = 1

        lanelet_start_angle = np.arctan2(
            -(lanelet.center_vertices[1][0] - lanelet.center_vertices[0][0]),
            (lanelet.center_vertices[1][1] - lanelet.center_vertices[0][1]),
        )
        lanelet_exit_angle = np.arctan2(
            -(lanelet.center_vertices[-1][0] - lanelet.center_vertices[-2][0]),
            (lanelet.center_vertices[-1][1] - lanelet.center_vertices[-2][1]),
        )
        # lanelet_full_angle = np.arctan2(
        #     -(lanelet.center_vertices[-1][0] - lanelet.center_vertices[0][0]),
        #     (lanelet.center_vertices[-1][1] - lanelet.center_vertices[0][1]),
        # )

        lanelet_width = norm(lanelet.left_vertices[-1] - lanelet.right_vertices[-1])
        diagonal_length = round((lanelet.distance[-1] ** 2 + lanelet_width ** 2) ** 0.5, decimal_precision)

        if not lanelet.successor:
            # if node_is_intersection[n] and lanelet.distance[-1] < INTERSECTION_OUTGOING_SINK_THRESHOLD:
            #     edge_lanelets[(n, node_mapping[-lanelet.lanelet_id])].add(lanelet.lanelet_id)
            #     discarded_nodes.add(n)
            #     continue
            if n != node_mapping[-lanelet.lanelet_id]:
                edges.append((n, node_mapping[-lanelet.lanelet_id]))
                edge_lanelets[(n, node_mapping[-lanelet.lanelet_id])].add(lanelet.lanelet_id)
                edge_attrs.append({
                    'weight': lanelet.distance[-1],
                    'lanelet_edge_type': LaneletEdgeType.SUCCESSOR,
                    'direction': 0,
                    'start_angle': lanelet_start_angle,
                    'exit_angle': lanelet_exit_angle,
                    'lanelet_id': lanelet.lanelet_id
                })

        for id_successor in lanelet.successor:
            successor_lanelet = lanelet_network.find_lanelet_by_id(id_successor)

            if validate:
                if lanelet.lanelet_id not in successor_lanelet.predecessor:
                    raise InvalidLaneletNetworkException("Lanelet inconsistency")
                if not check_connected(lanelet, successor_lanelet, atol=atol):
                    raise InvalidLaneletNetworkException("Lanelet inconsistency")

            if n != node_mapping[successor_lanelet.lanelet_id]:
                edges.append((n, node_mapping[successor_lanelet.lanelet_id]))
                edge_lanelets[(n, node_mapping[successor_lanelet.lanelet_id])].add(lanelet.lanelet_id)
                edge_attrs.append({
                    'weight': lanelet.distance[-1],
                    'lanelet_edge_type': LaneletEdgeType.SUCCESSOR,
                    'direction': 0,
                    'start_angle': lanelet_start_angle,
                    'exit_angle': lanelet_exit_angle,
                    'lanelet_id': lanelet.lanelet_id
                })

            # check for diagonal left succeeding lanelet
            if include_diagonals and successor_lanelet.adj_left and successor_lanelet.adj_left in lanelet_ids:
                adjacent_lanelet = lanelet_network.find_lanelet_by_id(successor_lanelet.adj_left)
                if lanelet.adj_left_same_direction:
                    edges.append((n, node_mapping[successor_lanelet.adj_left]))
                    edge_attrs.append({
                        'weight': diagonal_length,
                        'lanelet_edge_type': LaneletEdgeType.DIAGONAL_LEFT,
                        'direction': -1,
                        'start_angle': lanelet_start_angle,
                        'exit_angle': lanelet_exit_angle,
                        'lanelet_id': -1
                    })

                    if validate:
                        if adjacent_lanelet.adj_right != successor_lanelet.lanelet_id:
                            raise InvalidLaneletNetworkException("Lanelet inconsistency")
                        if not check_adjacent(adjacent_lanelet, successor_lanelet):
                            raise InvalidLaneletNetworkException("Lanelet inconsistency")
                # elif validate:
                #     if np.array_equal(lanelet.center_vertices[-1], adjacent_lanelet.center_vertices[0]):
                #         raise InvalidLaneletNetworkException("Lanelet inconsistency")
                #     # if not np.array_equal(lanelet.left_vertices[-1], adjacent_lanelet.right_vertices[0]):
                #     #     raise InvalidLaneletNetworkException("Lanelet inconsistency")

            # check for diagonal right succeeding lanelet
            if include_diagonals and successor_lanelet.adj_right and successor_lanelet.adj_right in lanelet_ids:
                adjacent_lanelet = lanelet_network.find_lanelet_by_id(successor_lanelet.adj_right)
                if lanelet.adj_right_same_direction:
                    edges.append((n, node_mapping[successor_lanelet.adj_right]))
                    edge_attrs.append({
                        'weight': diagonal_length,
                        'lanelet_edge_type': LaneletEdgeType.DIAGONAL_RIGHT,
                        'direction': 1,
                        'start_angle': lanelet_start_angle,
                        'exit_angle': lanelet_exit_angle,
                        'lanelet_id': -1
                    })

                    if validate:
                        if adjacent_lanelet.adj_left != successor_lanelet.lanelet_id:
                            raise InvalidLaneletNetworkException("Lanelet inconsistency")
                        if not check_adjacent(successor_lanelet, adjacent_lanelet):
                            raise InvalidLaneletNetworkException("Lanelet inconsistency")

                # elif validate:
                #     if np.array_equal(lanelet.center_vertices[-1], adjacent_lanelet.center_vertices[0]):
                #         raise InvalidLaneletNetworkException("Lanelet inconsistency")
                #     # if not np.array_equal(lanelet.right_vertices[-1], adjacent_lanelet.left_vertices[0]):
                #     #     raise InvalidLaneletNetworkException("Lanelet inconsistency")

        # check if succeeding lanelet of right lanelet (e.g. turning lane highway)
        if lanelet.adj_right and lanelet.adj_right in lanelet_ids:
            l_right = lanelet_network.find_lanelet_by_id(lanelet.adj_right)
            if include_diagonals and lanelet.adj_right_same_direction:
                if not check_adjacent(lanelet, l_right, raise_on_false=False):
                    if validate:
                        raise InvalidLaneletNetworkException("Lanelet inconsistency")
                        # if l_right.successor and not lanelet.successor:
                        #     raise InvalidLaneletNetworkException("Unexpected lane ending")
                        # if l_right.predecessor and not lanelet.predecessor:
                        #     raise InvalidLaneletNetworkException("Undefined lane beginning")
                else:
                    # check for diagonal right succeeding lanelet
                    for right_successor in l_right.successor:
                        # if not already in graph add it
                        if right_successor is not None and (lanelet.lanelet_id, right_successor) not in edges:
                            edges.append((n, node_mapping[right_successor]))
                            edge_attrs.append({
                                'weight': diagonal_length,
                                'lanelet_edge_type': LaneletEdgeType.DIAGONAL_RIGHT,
                                'direction': 1,
                                'start_angle': lanelet_start_angle,
                                'exit_angle': lanelet_exit_angle,
                                'lanelet_id': -1
                            })

                            if validate:
                                successor_lanelet = lanelet_network.find_lanelet_by_id(right_successor)
                                if l_right.lanelet_id not in successor_lanelet.predecessor:
                                    raise InvalidLaneletNetworkException("Lanelet inconsistency")
                                if not check_connected(l_right, successor_lanelet, atol=atol):
                                    raise InvalidLaneletNetworkException("Lanelet inconsistency")

            # elif validate:
            #     if np.array_equal(lanelet.center_vertices[0], l_right.center_vertices[-1]):
            #         raise InvalidLaneletNetworkException("Lanelet inconsistency")

            if include_adjacent:
                edges.append((n, node_mapping[l_right.lanelet_id]))
                lanelet_edge_type = LaneletEdgeType.ADJACENT_RIGHT if lanelet.adj_right_same_direction \
                    else LaneletEdgeType.OPPOSITE_RIGHT
                edge_attrs.append({
                    'weight': lanelet_width,
                    'lanelet_edge_type': lanelet_edge_type,
                    'direction': 1,
                    'start_angle': None,
                    'exit_angle': None,
                    'lanelet_id': -1
                })

        # check if succeeding lanelet of left lanelet (e.g. turning lane highway)
        if lanelet.adj_left and lanelet.adj_left in lanelet_ids:
            l_left = lanelet_network.find_lanelet_by_id(lanelet.adj_left)
            if include_diagonals and lanelet.adj_left_same_direction:
                if validate:
                    if not check_adjacent(l_left, lanelet):
                        raise InvalidLaneletNetworkException("Lanelet inconsistency")
                    # if l_left.successor and not lanelet.successor:
                    #     raise InvalidLaneletNetworkException("Unexpected lane ending")
                    # if l_left.predecessor and not lanelet.predecessor:
                    #     raise InvalidLaneletNetworkException("Undefined lane beginning")

                # check for diagonal left succeeding lanelet
                for left_successor in l_left.successor:

                    # if not already in graph add it
                    if left_successor is not None and (lanelet.lanelet_id, left_successor) not in edges:
                        edges.append((n, node_mapping[left_successor]))
                        edge_attrs.append({
                            'weight': diagonal_length,
                            'lanelet_edge_type': LaneletEdgeType.DIAGONAL_LEFT,
                            'direction': -1,
                            'start_angle': lanelet_start_angle,
                            'exit_angle': lanelet_exit_angle,
                            'lanelet_id': -1
                        })
                        if validate:
                            successor_lanelet = lanelet_network.find_lanelet_by_id(left_successor)
                            if l_left.lanelet_id not in successor_lanelet.predecessor:
                                raise InvalidLaneletNetworkException("Lanelet inconsistency")
                            if not check_connected(l_left, successor_lanelet, atol=atol):
                                raise InvalidLaneletNetworkException("Lanelet inconsistency")

            # elif validate:
            #     if np.array_equal(lanelet.center_vertices[0], l_left.center_vertices[-1]):
            #         raise InvalidLaneletNetworkException("Lanelet inconsistency")

            if include_adjacent:
                edges.append((n, node_mapping[l_left.lanelet_id]))
                lanelet_edge_type = LaneletEdgeType.ADJACENT_LEFT if lanelet.adj_left_same_direction \
                    else LaneletEdgeType.OPPOSITE_LEFT
                edge_attrs.append({
                    'weight': lanelet_width,
                    'lanelet_edge_type': lanelet_edge_type,
                    'direction': -1,
                    'start_angle': None,
                    'exit_angle': None,
                    'lanelet_id': -1
                })

    # assert len(node_pos) == len(set(node_mapping.values()))

    # add all nodes and edges to graph
    # edge_lanelets_included = set.union(*edge_lanelets.values())
    # missing_lanelets_1 = lanelet_ids.difference(edge_lanelets_included)
    # assert not missing_lanelets_1, missing_lanelets_1

    references_nodes = set(sum(edges, ()))
    edges_insert = []
    for i, e in enumerate(edges):
        if edge_attrs[i]['lanelet_id'] > -1:
            lanelet = lanelet_network.find_lanelet_by_id(edge_attrs[i]['lanelet_id'])
            if lanelet.adj_left_same_direction or lanelet.adj_right_same_direction:
                edge_attrs[i]['edge_cardinality'] = node_cardinality[node_mapping[edge_attrs[i]['lanelet_id']]]
            else:
                edge_attrs[i]['edge_cardinality'] = 1
        else:
            edge_attrs[i]['edge_cardinality'] = 0
        edges_insert.append([*e, edge_attrs[i]])
        if i > 0:
            assert set(edge_attrs[i].keys()) == set(edge_attrs[i - 1].keys())
    nodes_insert = []
    for n, p in node_pos.items():
        # lanelets = node_lanelets[n]
        # assert len(lanelets) > 0 and len(set(lanelets)) == len(lanelets)
        if n not in references_nodes:
            continue
        assert isinstance(n, int)

        lanelet_type = LaneletNodeType.LANELET
        if node_is_entry[n]:
            lanelet_type = LaneletNodeType.LANELET_BEGINNING
        if node_is_intersection[n]:
            lanelet_type = LaneletNodeType.INTERSECTION

        nodes_insert.append((n, dict(
            node_position=p,
            lanelet_type=lanelet_type,
            lanelets=tuple(node_lanelets[n])
        )))
        for e in edges:
            assert isinstance(e[0], int)
            assert isinstance(e[1], int)
            assert e[0] in node_pos
            assert e[1] in node_pos

    graph.add_nodes_from(nodes_insert)
    graph.add_edges_from(edges_insert)

    edge_lanelets_dict = {e: tuple(v) for e, v in edge_lanelets.items()}
    nx.set_edge_attributes(graph, edge_lanelets_dict, 'lanelets')

    # graph = remove_redundant_nodes(graph, iterations=-1)

    # edge_lanelets_attr = nx.get_edge_attributes(graph, 'lanelets')
    # edge_lanelets_included = set(sum(edge_lanelets_attr.values(), ()))
    # missing_lanelets = lanelet_ids.difference(edge_lanelets_included)
    # assert not missing_lanelets, missing_lanelets
    # if validate:
    #     if not nx.is_connected(graph.to_undirected()):
    #         raise InvalidLaneletNetworkException("Disconnected regions")

    return graph, lanelet_network
