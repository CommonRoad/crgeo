from typing import Optional, Tuple, Union

import logging
import math

import torch
import numpy as np

from commonroad.scenario.lanelet import Lanelet
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import resample_polyline


logger = logging.getLogger(__name__)


def resample_lanelet(lanelet: Lanelet, step=3.0):
    """
    Resamples the input polyline with the specified step size.

    The distances between each pair of consecutive vertices are examined. If it is larger than the step size,
    a new sample is added in between.

    :param polyline: polyline with 2D points
    :param step: minimum distance between each consecutive pairs of vertices
    :return: resampled polyline
    """
    polyline = lanelet.center_vertices
    if len(polyline) < 2:
        return np.array(polyline)

    polyline_new_c = [polyline[0]]
    polyline_new_r = [lanelet.right_vertices[0]]
    polyline_new_l = [lanelet.left_vertices[0]]

    current_idx = 0
    current_position = step
    current_distance = np.linalg.norm(polyline[0] - polyline[1])

    # iterate through all pairs of vertices of the polyline
    while current_idx < len(polyline) - 1:
        if current_position <= current_distance:
            # add new sample and increase current position
            ratio = current_position / current_distance
            polyline_new_c.append((1 - ratio) * polyline[current_idx] + ratio * polyline[current_idx + 1])
            polyline_new_r.append((1 - ratio) * lanelet.right_vertices[current_idx] +
                                  ratio * lanelet.right_vertices[current_idx + 1])
            polyline_new_l.append((1 - ratio) * lanelet.left_vertices[current_idx] +
                                  ratio * lanelet.left_vertices[current_idx + 1])
            current_position += step

        else:
            # move on to the next pair of vertices
            current_idx += 1
            # if we are out of vertices, then break
            if current_idx >= len(polyline) - 1:
                break
            # deduct the distance of previous vertices from the position
            current_position = current_position - current_distance
            # compute new distances of vertices
            current_distance = np.linalg.norm(polyline[current_idx + 1] - polyline[current_idx])

    # add the last vertex
    polyline_new_c.append(polyline[-1])
    polyline_new_r.append(lanelet.right_vertices[-1])
    polyline_new_l.append(lanelet.left_vertices[-1])

    lanelet._center_vertices = np.array(polyline_new_c).reshape([-1, 2])
    lanelet._right_vertices = np.array(polyline_new_r).reshape([-1, 2])
    lanelet._left_vertices = np.array(polyline_new_l).reshape([-1, 2])
    lanelet._distance = lanelet._compute_polyline_cumsum_dist([lanelet.center_vertices])



def get_v2l_edge_idx(
    lanelet_idx: int, ego_obstacle_idx: int, v2l_edge_index_t: torch.Tensor
) -> int:
    edge_idx_tuple = torch.tensor([ego_obstacle_idx, lanelet_idx])
    return (v2l_edge_index_t.H == edge_idx_tuple).all(dim=1).nonzero()[0].item()


def get_heading_error(v2l: dict, edge_idx: int) -> float:
    heading_error_idx = list(v2l.feature_columns).index("v2l_heading_error")
    return v2l["edge_attr"][edge_idx][heading_error_idx].item()


def get_lateral_error(v2l: dict, edge_idx: int) -> float:
    lateral_error_idx = list(v2l.feature_columns).index("v2l_lateral_error")
    return v2l["edge_attr"][edge_idx][lateral_error_idx].item()


def project_orthogonally(x1, x2, y):
    v = x2 - x1
    if np.dot(v, v) == 0:
        # logger.warning(f"[project_orthogonally] Points are the same ({x2} , {x1}).")
        return x1

    u = y - x1
    projection = np.dot(u, v) / np.dot(v, v) * v
    z = x1 + projection
    return z


def attach_vehicle_to_lanelet(
    position: np.ndarray, lanelet: Lanelet, arclength_abs: float
) -> Lanelet | None:
    # Vehicle exits this lanelet and vehicle center already passed last vertex.
    if arclength_abs >= lanelet.distance[-1]:
        return None

    # The center of the vehicle is somewhere on the lanelet.
    if arclength_abs > 0:
        cut_idx = np.where(lanelet.distance >= arclength_abs)[0][0] - 1
        remaining_arclength = arclength_abs - lanelet.distance[cut_idx]

        # partial_lanelet contains vehicles center and remaining lanelet.
        partial_lanelet = Lanelet(
            lanelet_id=1337,
            left_vertices=lanelet.left_vertices[cut_idx:],
            center_vertices=lanelet.center_vertices[cut_idx:],
            right_vertices=lanelet.right_vertices[cut_idx:],
            successor=lanelet.successor,
        )

        # TODO: Check if this really works inplace.
        resample_lanelet(partial_lanelet, 0.04)

        new_cut_idx = np.where(partial_lanelet.distance >= remaining_arclength)[0][0]

        left_vertices = partial_lanelet.left_vertices[new_cut_idx:]
        center_vertices = partial_lanelet.center_vertices[new_cut_idx:]
        right_vertices = partial_lanelet.right_vertices[new_cut_idx:]
        if lanelet_is_valid(left_vertices, center_vertices, right_vertices):
            return Lanelet(
                lanelet_id=1337,
                left_vertices=left_vertices,
                center_vertices=center_vertices,
                right_vertices=right_vertices,
                successor=lanelet.successor,
            )
        # else:
        #     logger.warning(
        #         "[attach_vehicle_to_lanelet] Invalid lanelet generated, Type 1."
        #     )

    # The vehicle has not entered the lanelet yet.
    if arclength_abs == 0:
        left_point = project_orthogonally(
            lanelet.left_vertices[0], lanelet.left_vertices[1], position
        )
        center_point = project_orthogonally(
            lanelet.center_vertices[0], lanelet.center_vertices[1], position
        )
        right_point = project_orthogonally(
            lanelet.right_vertices[0], lanelet.right_vertices[1], position
        )

        left_vertices = np.vstack((left_point, lanelet.left_vertices))
        center_vertices = np.vstack((center_point, lanelet.center_vertices))
        right_vertices = np.vstack((right_point, lanelet.right_vertices))
        if lanelet_is_valid(left_vertices, center_vertices, right_vertices):
            return Lanelet(
                lanelet_id=1337,
                left_vertices=left_vertices,
                center_vertices=center_vertices,
                right_vertices=right_vertices,
                successor=lanelet.successor
            )
        # else:
        #     logger.warning(
        #         "[attach_vehicle_to_lanelet] Invalid lanelet generated, Type 2."
        #     )


def point_on_line_at_distance(x1, x2, partial_distance):
    direction_vector = x2 - x1
    distance_x1_x2 = np.linalg.norm(direction_vector)
    t = partial_distance / distance_x1_x2
    return x1 + t * direction_vector


def create_final_point(vertices, cut_idx, remaining_distance):
    if cut_idx == len(vertices):
        cut_idx -= 1
    return point_on_line_at_distance(
        vertices[cut_idx - 1], vertices[cut_idx], remaining_distance
    )


def lanelet_is_valid(left_vertices, center_vertices, right_vertices) -> bool:
    l_dist = np.linalg.norm(np.diff(left_vertices, axis=0), axis=1)
    c_dist = np.linalg.norm(np.diff(center_vertices, axis=0), axis=1)
    r_dist = np.linalg.norm(np.diff(right_vertices, axis=0), axis=1)

    left_condition = sum(l_dist > 1e-3) >= 2 and sum(l_dist) > 0.04
    center_condition = sum(c_dist > 1e-3) >= 2 and sum(c_dist) > 0.04
    right_condition = sum(r_dist > 1e-3) >= 2 and sum(r_dist) > 0.04
    return left_condition and center_condition and right_condition


def create_inner_vehicle_polylines_from_lanelet_polylines(
    left_vertices: np.ndarray,
    center_vertices: np.ndarray,
    right_vertices: np.ndarray,
    width_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(left_vertices) == len(center_vertices) == len(right_vertices)

    ret_left = []
    ret_right = []
    for l, c, r in zip(left_vertices, center_vertices, right_vertices):
        direction_vec = np.array([r[0] - l[0], r[1] - l[1]])
        len_direction_vec = math.sqrt((direction_vec[0] ** 2) + (direction_vec[1] ** 2))
        unit_direction_vec = direction_vec / len_direction_vec
        ret_left.append(c + width_offset * unit_direction_vec)
        ret_right.append(c - width_offset * unit_direction_vec)

    return np.array(ret_left), np.array(ret_right)
