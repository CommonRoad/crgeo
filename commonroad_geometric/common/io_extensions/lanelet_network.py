import warnings
from queue import LifoQueue
from typing import Dict, List, Set, Tuple

import numpy as np
from commonroad.scenario.intersection import Intersection, IntersectionIncomingElement
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.geometry.helpers import cut_polylines_at_identical_segments, polyline_length


def find_lanelet_by_id(lanelet_network: LaneletNetwork, lanelet_id: int) -> Lanelet:
    """Finding lanelet without assertions or guards"""
    return lanelet_network._lanelets[lanelet_id]


def lanelet_orientation_at_position(lanelet: Lanelet, position: np.ndarray):
    """Approximates the lanelet orientation with the two closest point to the given state

    :param lanelet: Lanelet on which the orientation at the given state should be calculated
    :param position: Position where the lanelet's orientation should be calculated
    :return: An orientation in interval [-pi,pi]
    """
    center_vertices = lanelet.center_vertices

    position_diff = []
    for idx in range(len(center_vertices) - 1):
        vertex1 = center_vertices[idx]
        position_diff.append(np.linalg.norm(position - vertex1))

    closest_vertex_index = position_diff.index(min(position_diff))

    vertex1 = center_vertices[closest_vertex_index, :]
    vertex2 = center_vertices[closest_vertex_index + 1, :]
    direction_vector = vertex2 - vertex1

    # warnings.warn("Call to slow lanelet_orientation_at_position method. Consider using ContinuousPolyline.get_direction instead")

    return np.arctan2(direction_vector[1], direction_vector[0])



def collect_adjacent_lanelets(
    lanelet_network: LaneletNetwork,
    lanelet: Lanelet,
    include_left: bool = True,
    include_right: bool = True
) -> Tuple[List[Lanelet], List[Lanelet]]:
    found_lanelet_ids: Set[int] = set()
    adjacent_same_dir: List[Lanelet] = []
    adjacent_opposite_dir: List[Lanelet] = []
    new_adjacent_lanelets = { (lanelet.lanelet_id, True) }
    while new_adjacent_lanelets:
        lanelet_id, same_direction = new_adjacent_lanelets.pop()
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        found_lanelet_ids.add(lanelet_id)
        (adjacent_same_dir if same_direction else adjacent_opposite_dir).append(lanelet)

        if include_left:
            if lanelet.adj_left is not None and lanelet.adj_left not in found_lanelet_ids:
                adj_same_direction = not (same_direction ^ lanelet.adj_left_same_direction)
                new_adjacent_lanelets.add((lanelet.adj_left, adj_same_direction))
        if include_right:
            if lanelet.adj_right is not None and lanelet.adj_right not in found_lanelet_ids:
                adj_same_direction = not (same_direction ^ lanelet.adj_right_same_direction)
                new_adjacent_lanelets.add((lanelet.adj_right, adj_same_direction))

    return adjacent_same_dir, adjacent_opposite_dir


def segment_lanelets(
    scenario: Scenario,
    lanelet_max_segment_length: float,
    validate: bool = False
) -> LaneletNetwork:
    """Segment long lanelets into multiple smaller segments.

    This function ensures that all lanelets in the lanelet network are shorter than `lanelet_max_segment_length`.
    Lanelets which exceed `lanelet_max_segment_length` in length are segmented into multiple smaller segments.
    Their length is approximately in the interval `[lanelet_max_segment_length / 2, lanelet_max_segment_length]`.
    Adjacent lanelets are cut into segments of the same length.
    Predecessor, successor and adjacency relations are updated.
    """
    lanelet_network = scenario.lanelet_network
    start_id = scenario.generate_object_id()
    lanelets = sorted(lanelet_network.lanelets, key=lambda l: l.lanelet_id)
    all_lanelet_ids = set(lanelet.lanelet_id for lanelet in lanelet_network.lanelets)
    lanelet_ids_to_be_removed = []

    while lanelets:
        # group adjacent lanelets
        adjacent_same_dir, adjacent_opposite_dir = collect_adjacent_lanelets(lanelet_network, lanelets[0])
        # adjacent_same_dir contains lanelets[0]
        adjacent_lanelets = sorted(adjacent_same_dir + adjacent_opposite_dir, key=lambda l: l.lanelet_id)
        all_new_lanelet_segments = []

        longest_lanelet = max(adjacent_lanelets, key=lambda l: l.distance[-1])
        if longest_lanelet.distance[-1] > lanelet_max_segment_length:
            # at least one lanelet in the group is too long
            # -> segment all equally into multiple shorter lanelets
            num_segments = int(np.ceil(longest_lanelet.distance[-1] / lanelet_max_segment_length))
            old_lanelet_id_to_lanelet_segment_ids = {}

            for lanelet in adjacent_lanelets:
                segment_lanelet_ids = start_id + 1 + np.arange(num_segments, dtype=int)
                old_lanelet_id_to_lanelet_segment_ids[lanelet.lanelet_id] = segment_lanelet_ids
                try:
                    all_lanelet_ids.remove(lanelet.lanelet_id)
                except KeyError:
                    warnings.warn("Unexpected KeyError from 'all_lanelet_ids.remove(lanelet.lanelet_id)'")
                all_lanelet_ids.update(segment_lanelet_ids)
                start_id += num_segments

                # If we cut center, left and right vertices independently at the same distances along their polyline
                # then the cuts will sometimes (e.g. in bends) be at different polyline segments.
                # In these cases the resulting smaller polylines do not all have the same amount of vertices.
                # This can cause problems down the line.
                # Here we ensure that the polylines are cut into segments which all have the same amount of
                # vertices.
                center_segment_length = polyline_length(lanelet.center_vertices) / num_segments
                center_cut_distances = np.arange(1, num_segments, dtype=float) * center_segment_length
                center_vertices, left_vertices, right_vertices = cut_polylines_at_identical_segments(
                    lines=[ lanelet.center_vertices, lanelet.left_vertices, lanelet.right_vertices ],
                    distance=center_cut_distances,
                )
                assert all(
                    left_vertices[i].shape == center_vertices[i].shape == right_vertices[i].shape
                    for i in range(num_segments)
                )

                lanelet_segments = [
                    clone_lanelet(
                        lanelet,
                        lanelet_id=int(segment_lanelet_ids[i]),
                        left_vertices=left_vertices[i],
                        center_vertices=center_vertices[i],
                        right_vertices=right_vertices[i],
                        predecessor=[ int(segment_lanelet_ids[i - 1]) ] if i > 0 else lanelet.predecessor,
                        successor=[ int(segment_lanelet_ids[i + 1]) ] if i < num_segments - 1 else lanelet.successor,
                    )
                    for i in range(num_segments)
                ]
                all_new_lanelet_segments.extend((
                    (i, segment)
                    for i, segment in enumerate(lanelet_segments)
                ))

                # update predecessor and successor relations
                for predecessor_id in lanelet.predecessor:
                    predecessor = lanelet_network.find_lanelet_by_id(predecessor_id)
                    if validate:
                        assert lanelet.lanelet_id in predecessor.successor
                    idx = predecessor.successor.index(lanelet.lanelet_id)
                    predecessor.successor[idx] = segment_lanelet_ids[0]
                for successor_id in lanelet.successor:
                    successor = lanelet_network.find_lanelet_by_id(successor_id)
                    if validate:
                        lanelet.lanelet_id in successor.predecessor
                    idx = successor.predecessor.index(lanelet.lanelet_id)
                    successor.predecessor[idx] = segment_lanelet_ids[-1]

                # add new lanelets to lanelet network
                for i, lanelet_segment in enumerate(lanelet_segments):
                    lanelet_network.add_lanelet(lanelet_segment, rtree=i == len(lanelet_segments) - 1)

                # postpone removing the replaced lanelet because removing it from the lanelet network also
                # removes any (e.g. adjacency) relations for this lanelet
                lanelet_ids_to_be_removed.append(lanelet.lanelet_id)

            # update adjacency relations
            for i, lanelet in all_new_lanelet_segments:
                if lanelet.adj_left is not None:
                    if lanelet.adj_left_same_direction:
                        lanelet._adj_left = old_lanelet_id_to_lanelet_segment_ids[lanelet.adj_left][i]
                    else:
                        lanelet._adj_left = old_lanelet_id_to_lanelet_segment_ids[lanelet.adj_left][num_segments - 1 - i]
                if lanelet.adj_right is not None:
                    if lanelet.adj_right_same_direction:
                        lanelet._adj_right = old_lanelet_id_to_lanelet_segment_ids[lanelet.adj_right][i]
                    else:
                        lanelet._adj_right = old_lanelet_id_to_lanelet_segment_ids[lanelet.adj_right][num_segments - 1 - i]

        adjacent_lanelet_ids = {lanelet.lanelet_id for lanelet in adjacent_lanelets}
        lanelets = [lanelet for lanelet in lanelets if lanelet.lanelet_id not in adjacent_lanelet_ids]

    # remove replaced lanelets from lanelet network
    for lanelet_id in lanelet_ids_to_be_removed:
        lanelet_network.remove_lanelet(lanelet_id)

    return lanelet_network


def check_connected(
    lane_1: Lanelet,
    lane_2: Lanelet,
    atol: float = 0.0,
    raise_on_false: bool = True
) -> bool:
    result = all([
        np.allclose(lane_1.center_vertices[-1], lane_2.center_vertices[0], rtol=0.0, atol=atol),
        np.allclose(lane_1.left_vertices[-1], lane_2.left_vertices[0], rtol=0.0, atol=atol),
        np.allclose(lane_1.right_vertices[-1], lane_2.right_vertices[0], rtol=0.0, atol=atol),
    ])
    if raise_on_false and not result:
        raise ValueError("Lanelets not connected")
    return result


def check_adjacent(
    lane_left: Lanelet,
    lane_right: Lanelet,
    raise_on_false: bool = True
) -> bool:
    if lane_left.right_vertices.shape != lane_right.left_vertices.shape or \
        lane_left.center_vertices.shape != lane_right.center_vertices.shape:
        if raise_on_false:
            raise ValueError("Lanelets not conneced")
        return False
    result = all([
        np.allclose(lane_left.right_vertices, lane_right.left_vertices),
        not np.isclose(lane_left.center_vertices, lane_right.center_vertices).any(),
        not np.isclose(lane_left.center_vertices, lane_right.center_vertices).any(),
    ])
    if raise_on_false and not result:
        raise ValueError("Lanelets not connected")
    return result


def copy_lanelet_with_precision(
    lanelet: Lanelet,
    decimal_precision: int
) -> Lanelet:
    return Lanelet(
        lanelet_id=lanelet.lanelet_id,
        left_vertices=np.around(lanelet.left_vertices, decimal_precision),
        center_vertices=np.around(lanelet.center_vertices, decimal_precision),
        right_vertices=np.around(lanelet.right_vertices, decimal_precision),
        predecessor=lanelet.predecessor,
        successor=lanelet.successor,
        adjacent_left=lanelet.adj_left,
        adjacent_left_same_direction=lanelet.adj_left_same_direction,
        adjacent_right=lanelet.adj_right,
        adjacent_right_same_direction=lanelet.adj_right_same_direction,
        line_marking_left_vertices=lanelet.line_marking_left_vertices,
        line_marking_right_vertices=lanelet.line_marking_right_vertices,
        stop_line=lanelet.stop_line,
        lanelet_type=lanelet.lanelet_type,
        user_one_way=lanelet.user_one_way,
        user_bidirectional=lanelet.user_bidirectional,
        traffic_signs=lanelet.traffic_signs,
        traffic_lights=lanelet.traffic_lights,
    )


def clone_lanelet(
    lanelet: Lanelet,
    **kwargs
) -> Lanelet:
    lanelet_kwargs = dict(
        lanelet_id=lanelet.lanelet_id,
        left_vertices=lanelet.left_vertices.copy() if lanelet.left_vertices is not None else None,
        center_vertices=lanelet.center_vertices.copy() if lanelet.center_vertices is not None else None,
        right_vertices=lanelet.right_vertices.copy() if lanelet.right_vertices is not None else None,
        predecessor=lanelet.predecessor,
        successor=lanelet.successor,
        adjacent_left=lanelet.adj_left,
        adjacent_left_same_direction=lanelet.adj_left_same_direction,
        adjacent_right=lanelet.adj_right,
        adjacent_right_same_direction=lanelet.adj_right_same_direction,
        line_marking_left_vertices=lanelet.line_marking_left_vertices,
        line_marking_right_vertices=lanelet.line_marking_right_vertices,
        stop_line=lanelet.stop_line,
        lanelet_type=lanelet.lanelet_type,
        user_one_way=lanelet.user_one_way,
        user_bidirectional=lanelet.user_bidirectional,
        traffic_signs=lanelet.traffic_signs,
        traffic_lights=lanelet.traffic_lights,
    )
    lanelet_kwargs.update(kwargs)
    return Lanelet(**lanelet_kwargs)

def get_intersection_successors(
    intersection: Intersection,
    include_straight: bool = True,
    include_left: bool = True,
    include_right: bool = True,
) -> Set[int]:
    includes = []
    for incoming_element in intersection.incomings:
        if include_straight:
            includes.append(incoming_element.successors_straight)
        if include_left:
            includes.append(incoming_element.successors_left)
        if include_right:
            includes.append(incoming_element.successors_right)
    successor_set = set.union(*includes)
    return successor_set

def map_out_lanelets_to_intersections(
    lanelet_network: LaneletNetwork,
    include_straight: bool = True,
    include_left: bool = True,
    include_right: bool = True,
    include_neighbors: bool = True
) -> Dict[int, Intersection]:
    intersection_outgoing_map: Dict[int, Intersection] = {}

    for intersection in lanelet_network.intersections:
        for incoming_element in intersection.incomings:
            includes = []
            if include_straight:
                includes.append(incoming_element.successors_straight)
            if include_left:
                includes.append(incoming_element.successors_left)
            if include_right:
                includes.append(incoming_element.successors_right)
            successor_set = set.union(*includes)
            
            for successor_id in successor_set:
                successor = lanelet_network.find_lanelet_by_id(successor_id)
                if successor is None:
                    continue
                #assert successor.distance[-1] < 50
                # assert successor_id not in intersection_outgoing_map
                if successor_id in intersection_outgoing_map:
                    continue
                intersection_outgoing_map[successor_id] = intersection
                
                if include_neighbors:
                    for neighbor in get_neighbors_all(lanelet_network, successor):
                        if neighbor not in successor_set:
                            intersection_outgoing_map[neighbor] = intersection

    return intersection_outgoing_map

def map_successor_lanelets_to_intersections(
    lanelet_network: LaneletNetwork,
    include_straight: bool = True,
    include_left: bool = True,
    include_right: bool = True,
    validate: bool = False
) -> Dict[int, Intersection]:
    intersection_outgoing_map: Dict[int, Intersection] = {}

    for intersection in lanelet_network.intersections:
        for incoming_element in intersection.incomings:
            includes = []
            if include_straight:
                includes.append(incoming_element.successors_straight)
            if include_left:
                includes.append(incoming_element.successors_left)
            if include_right:
                includes.append(incoming_element.successors_right)
            successor_set = set.union(*includes)
            
            for successor_id in successor_set:
                successor = lanelet_network.find_lanelet_by_id(successor_id)
                if successor is None:
                    continue
                for successor_successor_id in successor.successor:
                    if validate:
                        assert successor_successor_id not in intersection_outgoing_map or intersection_outgoing_map[successor_successor_id].intersection_id == intersection.intersection_id
                    intersection_outgoing_map[successor_successor_id] = intersection
                        
    return intersection_outgoing_map


def merge_successors(
    lanelet_network: LaneletNetwork,
    max_iterations: int = -1,
    ignore_multilane: bool = False,
    validate: bool = False
) -> List[Dict[int, int]]:

    if max_iterations < 0:
        max_iterations = np.inf

    iter = 0
    dead_state_counter = 0
    deleted_all: List[Dict[int, int]] = []
    while iter < max_iterations:
        intersection_incoming_map = lanelet_network.map_inc_lanelets_to_intersections
        intersection_outgoing_map = map_out_lanelets_to_intersections(lanelet_network)
        intersection_outgoing_lanelets = set(intersection_outgoing_map.keys())
        # intersection_lanelets = set.union(
        #     set(intersection_incoming_map.keys()),
        #     set(intersection_outgoing_map.keys())
        # )

        deleted_iter: Dict[int, int] = {}
        replaced: Set[int] = set()

        processing_queue = LifoQueue()
        for i, lanelet in enumerate(lanelet_network.lanelets):
            processing_queue.put(lanelet.lanelet_id)

        while not processing_queue.empty():
            lanelet_id = processing_queue.get()
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)

            already_processed = set.union(
                set(deleted_iter.keys()),
                replaced
            )

            if any((
                len(lanelet.predecessor) != 1,
                #len(lanelet.successor) > 1,
                ignore_multilane and lanelet.adj_left_same_direction,
                ignore_multilane and lanelet.adj_right_same_direction,
                lanelet.lanelet_id in intersection_outgoing_map,
                lanelet.lanelet_id in already_processed,
                any((n in already_processed for n in lanelet.predecessor)),
            )):
                continue

            if validate:
                assert lanelet.predecessor[0] not in intersection_incoming_map

            predecessor = lanelet_network.find_lanelet_by_id(lanelet.predecessor[0])
            if lanelet.predecessor[0] in intersection_incoming_map:
                continue
            if any((
                lanelet.adj_left and not lanelet.adj_left_same_direction and not predecessor.adj_left,
                len(predecessor.successor) != 1, # and predecessor.lanelet_id not in intersection_outgoing_lanelets,
                ignore_multilane and predecessor.adj_left_same_direction,
                ignore_multilane and predecessor.adj_right_same_direction,
                predecessor.lanelet_id in intersection_outgoing_lanelets,
                #not lanelet.adj_right_same_direction and not lanelet.adj_left_same_direction and len(get_successor_predecessors(lanelet_network, lanelet)) > 1,
                #not predecessor.adj_right_same_direction and not predecessor.adj_left_same_direction and get_neighbors_all(lanelet_network, lanelet),
                #has_successor and len(successor.predecessor) != 1 and successor.lanelet_id not in intersection_outgoing_lanelets,
                #has_successor and ignore_multilane and successor.adj_left_same_direction,
                #has_successor and ignore_multilane and successor.adj_right_same_direction,
                any((n in already_processed for n in predecessor.predecessor)),
                #has_successor and any((n in already_processed for n in successor.successor)),
                any((n in already_processed for n in predecessor.successor)),
                #lanelet.adj_left_same_direction == False and len(get_neighbors_right(lanelet_network, lanelet_network.find_lanelet_by_id(lanelet.adj_left))) > 0,
                #predecessor.adj_left_same_direction == False and len(get_neighbors_right(lanelet_network, lanelet_network.find_lanelet_by_id(predecessor.adj_left))) > 0,
                #has_successor and any((n in already_processed for n in successor.predecessor)),
            )):
                continue

            # avoiding splitting lanes
            cancel_merge = False
            tmp_lanelet = lanelet
            adj_set: Set[int] = set()
            successor_set: Set[int] = set()
            while tmp_lanelet.adj_right_same_direction:
                adj_set.add(tmp_lanelet.adj_right)
                successor_set.update(tmp_lanelet.successor)
                adj_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_right)
                if len(adj_lanelet.predecessor) != 1:
                    break
                tmp_predecessor = lanelet_network.find_lanelet_by_id(tmp_lanelet.predecessor[0])
                adj_lanelet_predecessor = lanelet_network.find_lanelet_by_id(adj_lanelet.predecessor[0])
                if tmp_predecessor.adj_right != adj_lanelet_predecessor.lanelet_id:
                    cancel_merge = True
                    break
                if len(adj_lanelet_predecessor.successor) != 1:
                    cancel_merge = True
                    break
                if len(adj_lanelet_predecessor.predecessor) == 1:
                    adj_lanelet_pre_predecessor = lanelet_network.find_lanelet_by_id(adj_lanelet_predecessor.predecessor[0])
                    if len(adj_lanelet_pre_predecessor.successor) != 1:
                        cancel_merge = True
                        break
                tmp_lanelet = adj_lanelet
            if cancel_merge:
                continue

            cancel_merge = False
            tmp_lanelet = lanelet
            while tmp_lanelet.adj_left_same_direction:
                adj_set.add(tmp_lanelet.adj_right)
                successor_set.update(tmp_lanelet.successor)
                adj_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_left)
                if len(adj_lanelet.predecessor) != 1:
                    break
                adj_lanelet_predecessor = lanelet_network.find_lanelet_by_id(adj_lanelet.predecessor[0])
                if predecessor.adj_left != adj_lanelet_predecessor.lanelet_id:
                    cancel_merge = True
                    break
                if len(adj_lanelet_predecessor.successor) != 1:
                    cancel_merge = True
                    break
                if len(adj_lanelet_predecessor.predecessor) == 1:
                    adj_lanelet_pre_predecessor = lanelet_network.find_lanelet_by_id(adj_lanelet_predecessor.predecessor[0])
                    if len(adj_lanelet_pre_predecessor.successor) != 1:
                        cancel_merge = True
                        break
                tmp_lanelet = adj_lanelet
            if cancel_merge:
                continue

            if validate:
                assert predecessor.successor == [lanelet.lanelet_id]
            # if has_successor:
            #     assert successor.predecessor == [lanelet.lanelet_id]
            if validate:
                assert check_connected(predecessor, lanelet, atol=1e-1)
            # if has_successor:
            #     assert check_connected(lanelet, successor, atol=1e-1)

            left_vertices = np.vstack([
                predecessor.left_vertices[:-1],
                lanelet.left_vertices
            ])
            center_vertices = np.vstack([
                predecessor.center_vertices[:-1],
                lanelet.center_vertices
            ])
            right_vertices = np.vstack([
                predecessor.right_vertices[:-1],
                lanelet.right_vertices
            ])

            lanelet._center_vertices = center_vertices
            lanelet._right_vertices = right_vertices
            lanelet._left_vertices = left_vertices
            for n in predecessor.predecessor:
                if validate:
                    assert n != lanelet.lanelet_id
                lanelet.add_predecessor(n)
                if validate:
                    assert check_connected(
                        lanelet_network.find_lanelet_by_id(n),
                        lanelet,
                        atol=1e-1
                    )

            replaced.add(lanelet.lanelet_id)
            deleted_iter[predecessor.lanelet_id] = lanelet.lanelet_id

            if lanelet.adj_left_same_direction:
                processing_queue.put(lanelet.adj_left)
            if lanelet.adj_right_same_direction:
                processing_queue.put(lanelet.adj_right)

        for lanelet in lanelet_network.lanelets:
            for k, v in deleted_iter.items():
                if lanelet.lanelet_id in {k, v}:
                    continue
                if k in lanelet.predecessor:
                    lanelet.remove_predecessor(k)
                    lanelet.add_predecessor(v)
                    if validate:
                        assert v != lanelet.lanelet_id
                if k in lanelet.successor:
                    lanelet.remove_successor(k)
                    lanelet.add_successor(v)
                    if validate:
                        assert v != lanelet.lanelet_id
                if lanelet.adj_left == k:
                    lanelet._adj_left = v
                    assert v != lanelet.lanelet_id
                if lanelet.adj_right == k:
                    lanelet._adj_right = v
                    if validate:
                        assert v != lanelet.lanelet_id

        for intersection in lanelet_network.intersections:
            for incoming_element in intersection.incomings:
                for k, v in deleted_iter.items():
                    if k in incoming_element.incoming_lanelets:
                        incoming_element.incoming_lanelets.discard(k)
                        incoming_element.incoming_lanelets.add(v)

                    if k in incoming_element.successors_left:
                        incoming_element.successors_left.discard(k)
                        incoming_element.successors_left.add(v)

                    if k in incoming_element.successors_straight:
                        incoming_element.successors_straight.discard(k)
                        incoming_element.successors_straight.add(v)

                    if k in incoming_element.successors_right:
                        incoming_element.successors_right.discard(k)
                        incoming_element.successors_right.add(v)

        for k, v in deleted_iter.items():
            lanelet_network.remove_lanelet(k)
        
        for lanelet in lanelet_network.lanelets:
            for x_id in lanelet.predecessor:
                x = lanelet_network.find_lanelet_by_id(x_id)
                if validate:
                    assert lanelet.lanelet_id in x.successor
            for x_id in lanelet.successor:
                x = lanelet_network.find_lanelet_by_id(x_id)
                if validate:
                    assert lanelet.lanelet_id in x.predecessor

        lanelet_network.cleanup_lanelet_references()
        if not deleted_iter:
            dead_state_counter += 1
            if dead_state_counter > 1:
                break
        else:
            dead_state_counter = 0
        deleted_all.append(deleted_iter)
        iter += 1
    
    return deleted_all


def remove_unused_traffic_signs_and_lights(scenario: Scenario):
    all_traffic_signs = {x.traffic_sign_id for x in scenario.lanelet_network.traffic_signs}
    all_traffic_lights = {x.traffic_light_id for x in scenario.lanelet_network.traffic_lights}
    references_traffic_signs = set()
    references_traffic_lights = set()
    for lanelet in scenario.lanelet_network.lanelets:
        lanelet._traffic_signs = set.intersection(all_traffic_signs, lanelet.traffic_signs)
        lanelet._traffic_lights = set.intersection(all_traffic_lights, lanelet.traffic_lights)
        references_traffic_signs.update(lanelet.traffic_signs)
        references_traffic_lights.update(lanelet.traffic_lights)
    for x in scenario.lanelet_network.traffic_lights:
        if x.traffic_light_id not in references_traffic_lights:
            scenario.remove_traffic_light(x)
    for x in scenario.lanelet_network.traffic_signs:
        if x.traffic_sign_id not in references_traffic_signs:
            scenario.remove_traffic_sign(x)


def cleanup_intersections(
    lanelet_network: LaneletNetwork
) -> LaneletNetwork:

    #intersection_outgoing_map = map_out_lanelets_to_intersections(lanelet_network)
    intersection_successor_map = map_successor_lanelets_to_intersections(lanelet_network)

    deleted_iter: Set[int] = set()

    for lanelet in lanelet_network.lanelets:
        # if len(lanelet.predecessor) > 1:
        #     do_cleanup = False
        #     to_delete = set()
        #     for predecessor in lanelet.predecessor:
        #         if predecessor in intersection_outgoing_map:
        #             do_cleanup = True
        #         else:
        #             to_delete.add(predecessor)
        #     if do_cleanup:
        #         deleted_iter.update(to_delete)

        if len(lanelet.predecessor) == 0:
            do_cleanup = True
            for successor_id in lanelet.successor:
                if successor_id not in intersection_successor_map:
                    do_cleanup = False
            if do_cleanup:
                deleted_iter.update({lanelet.lanelet_id})

    for lanelet_id in deleted_iter:
        lanelet_network.remove_lanelet(lanelet_id)

    remove_empty_intersections(lanelet_network, strict=False)

    return deleted_iter


def remove_empty_intersections(lanelet_network: LaneletNetwork, strict: bool = True):
    # removing empty incoming intersection elements
    lanelet_ids = {l.lanelet_id for l in lanelet_network.lanelets}

    for intersection in lanelet_network.intersections:
        new_incomings: List[IntersectionIncomingElement] = []
        left_of_mapping: Dict[int, int] = {}
        for incoming_element in intersection.incomings:
            incoming_element._incoming_lanelets = {x for x in incoming_element._incoming_lanelets if x in lanelet_ids}
            incoming_element._successors_left = {x for x in incoming_element._successors_left if x in lanelet_ids}
            incoming_element._successors_right = {x for x in incoming_element._successors_right if x in lanelet_ids}
            incoming_element._successors_straight = {x for x in incoming_element._successors_straight if x in lanelet_ids}
            has_predecessor = len(set.union(
                incoming_element._successors_left,
                incoming_element._successors_right,
                incoming_element._successors_straight
            )) > 0
            if (not strict or has_predecessor) and len(incoming_element.incoming_lanelets) > 0:
                new_incomings.append(incoming_element)
            else:
                left_of_mapping[incoming_element.incoming_id] = incoming_element.left_of
        for incoming_element in new_incomings:
            if incoming_element.left_of:
                if incoming_element.left_of in left_of_mapping:
                    incoming_element._left_of = left_of_mapping[incoming_element.left_of]

        if len(new_incomings) > 0:
            intersection._incomings = new_incomings
        elif strict:
            lanelet_network.remove_intersection(intersection.intersection_id)


def get_neighbors_left(lanelet_network: LaneletNetwork, lanelet: Lanelet, validate: bool = False) -> Set[int]:
    neighbors = set()
    if lanelet.adj_left_same_direction:
        neighbors.add(lanelet.adj_left)
    if len(lanelet.predecessor) == 1:
        predecessor = lanelet_network.find_lanelet_by_id(lanelet.predecessor[0])
        for successor in predecessor.successor:
            if successor != lanelet.lanelet_id:
                neighbors.add(successor)
        if predecessor.adj_left_same_direction:
            for successor in lanelet_network.find_lanelet_by_id(predecessor.adj_left).successor:
                if successor != lanelet.lanelet_id:
                    neighbors.add(successor)
    if validate:
        assert lanelet.lanelet_id not in neighbors
    return neighbors


def get_neighbors_right(lanelet_network: LaneletNetwork, lanelet: Lanelet, validate: bool = False) -> Set[int]:
    neighbors = set()
    if lanelet.adj_right_same_direction:
        neighbors.add(lanelet.adj_right)
    if len(lanelet.predecessor) == 1:
        predecessor = lanelet_network.find_lanelet_by_id(lanelet.predecessor[0])
        for successor in predecessor.successor:
            if successor != lanelet.lanelet_id:
                neighbors.add(successor)
        if predecessor.adj_right_same_direction:
            for successor in lanelet_network.find_lanelet_by_id(predecessor.adj_right).successor:
                if successor != lanelet.lanelet_id:
                    neighbors.add(successor)
    if validate:
        assert lanelet.lanelet_id not in neighbors
    return neighbors


def get_neighbors_all(lanelet_network: LaneletNetwork, lanelet: Lanelet) -> Set[int]:
    neighbors = set.union(
        get_neighbors_left(lanelet_network, lanelet),
        get_neighbors_right(lanelet_network,  lanelet)
    )
    return neighbors


def get_successor_predecessors(lanelet_network: LaneletNetwork, lanelet: Lanelet) -> Set[int]:
    predecessors = set()
    for successor in lanelet.successor:
        for predecessor in lanelet_network.find_lanelet_by_id(successor).predecessor:
            predecessors.add(predecessor)
    return predecessors


def map_lanelets_to_adjacency_clusters(
    lanelet_network: LaneletNetwork,
    decimal_precision: int = 0,
    validate: bool = False
) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]]]:
    adjacency_map: Dict[int, Set[int]] = {}
    cluster_pos_map: Dict[int, np.ndarray] = {}
    processed: Set[int] = set()

    intersection_outgoing_map = map_out_lanelets_to_intersections(
        lanelet_network,
        include_straight=False
    )

    for lanelet in sorted(lanelet_network.lanelets, key=lambda x: x.lanelet_id):
        if lanelet.lanelet_id in intersection_outgoing_map:
            continue
        if lanelet.lanelet_id in adjacency_map:
            continue
        if lanelet.lanelet_id in processed:
            continue

        neighbors_left = get_neighbors_left(lanelet_network, lanelet)
        neighbors_right = get_neighbors_right(lanelet_network, lanelet)

        if neighbors_left:
            continue

        cluster_pos_map[lanelet.lanelet_id] = lanelet.left_vertices[0].round(decimal_precision)

        adjacency_map[lanelet.lanelet_id] = {lanelet.lanelet_id}
        for predecessor_id in lanelet.predecessor:
            adjacency_map[lanelet.lanelet_id].add(-predecessor_id)

        queue = LifoQueue()
        to_process: Set[int] = set()

        for neighbor in neighbors_right:
            queue.put(neighbor)
            to_process.add(neighbor)

        processed.add(lanelet.lanelet_id)
        while not queue.empty():
            adj_lanelet_id = queue.get()
            if validate:
                assert adj_lanelet_id is not None
            if adj_lanelet_id in processed:
                continue
            if adj_lanelet_id in adjacency_map:
                continue
                # TODO warn

            adj_lanelet = lanelet_network.find_lanelet_by_id(adj_lanelet_id)
            neighbors_right = get_neighbors_right(lanelet_network, adj_lanelet)
            adjacency_map[lanelet.lanelet_id].add(adj_lanelet_id)
            for predecessor_id in adj_lanelet.predecessor:
                adjacency_map[lanelet.lanelet_id].add(-predecessor_id)

            for neighbor in neighbors_right:
                queue.put(neighbor)
                to_process.add(neighbor)

            processed.add(adj_lanelet_id)

    for lanelet in sorted(lanelet_network.lanelets, key=lambda x: x.lanelet_id):
        if lanelet.successor:
            continue
        if -lanelet.lanelet_id in intersection_outgoing_map:
            continue
        if -lanelet.lanelet_id in adjacency_map:
            continue
        if -lanelet.lanelet_id in processed:
            continue

        neighbors_left = get_neighbors_left(lanelet_network, lanelet)
        neighbors_right = get_neighbors_right(lanelet_network, lanelet)

        if neighbors_left:
            continue

        cluster_pos_map[-lanelet.lanelet_id] = lanelet.left_vertices[-1].round(decimal_precision)

        adjacency_map[-lanelet.lanelet_id] = {-lanelet.lanelet_id}

        queue = LifoQueue()
        to_process: Set[int] = set()

        for neighbor in neighbors_right:
            queue.put(neighbor)
            to_process.add(neighbor)

        processed.add(-lanelet.lanelet_id)
        while not queue.empty():
            adj_lanelet_id = queue.get()
            if validate:
                assert adj_lanelet_id is not None
            if -adj_lanelet_id in processed:
                continue
            if -adj_lanelet_id in adjacency_map:
                continue
                # TODO warn

            adj_lanelet = lanelet_network.find_lanelet_by_id(adj_lanelet_id)
            neighbors_right = get_neighbors_right(lanelet_network, adj_lanelet)
            adjacency_map[-lanelet.lanelet_id].add(-adj_lanelet_id)

            for neighbor in neighbors_right:
                queue.put(neighbor)
                to_process.add(neighbor)

            processed.add(-adj_lanelet_id)

    return adjacency_map, cluster_pos_map


def remove_unconnected_lanelets(lanelet_network: LaneletNetwork) -> LaneletNetwork:
    if len(lanelet_network.lanelets) == 0:
        return lanelet_network
    all_lanelet_ids = set(l.lanelet_id for l in lanelet_network.lanelets)
    referenced_lanelet_ids = set.union(*[set(l.successor + l.predecessor) for l in lanelet_network.lanelets])
    unconnected_lanelet_ids = all_lanelet_ids - referenced_lanelet_ids

    for lanelet_id in unconnected_lanelet_ids:
        lanelet_network.remove_lanelet(lanelet_id)
    lanelet_network.cleanup_lanelet_references()
