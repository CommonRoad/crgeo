import warnings
from typing import List

import numpy as np
from commonroad.geometry.shape import Rectangle, Shape
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario


def relative_orientation(angle_1, angle_2):
    """Computes the angle between two angles."""

    phi = (angle_2 - angle_1) % (2 * np.pi)
    if phi > np.pi:
        phi -= (2 * np.pi)

    return phi


def chaikins_corner_cutting(polyline: np.ndarray, num_refinements: int = 4) -> np.ndarray:
    """Chaikin's corner cutting algorithm.

    Chaikin's corner cutting algorithm smooths a polyline by replacing each original point with two new points.
    The new points are at 1/4 and 3/4 along the way of an edge.

    :param polyline: polyline with 2D points
    :param num_refinements: how many times to apply the chaikins corner cutting algorithm. setting to 6 is smooth enough
                            for most cases
    :return: smoothed polyline
    """
    for _ in range(num_refinements):
        L = polyline.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        polyline = L * 0.75 + R * 0.25

    return polyline


def compute_polyline_length(polyline: np.ndarray) -> float:
    """Computes the path length of a given polyline.

    :param polyline: The polyline
    :return: The path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for path length computation p={}'.format(polyline)

    distance_between_points = np.diff(polyline, axis=0)
    # noinspection PyTypeChecker
    return np.sum(np.sqrt(np.sum(distance_between_points ** 2, axis=1)))


def resample_polyline_with_length_check(polyline, step=2):
    """ Resamples polyline with length check."""
    length = np.linalg.norm(polyline[-1] - polyline[0])
    if length > step:
        polyline = resample_polyline(polyline, step)
    else:
        polyline = resample_polyline(polyline, length / 10.0)

    return polyline


def resample_polyline(polyline: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resamples the input polyline with the specified step size.

    The distances between each pair of consecutive vertices are examined. If it is larger than the step size,
    a new sample is added in between.

    :param polyline: polyline with 2D points
    :param step: minimum distance between each consecutive pairs of vertices
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return np.array(polyline)

    polyline_new = [polyline[0]]

    current_idx = 0
    current_position = step
    current_distance = np.linalg.norm(polyline[0] - polyline[1])

    # iterate through all pairs of vertices of the polyline
    while current_idx < len(polyline) - 1:
        if current_position <= current_distance:
            # add new sample and increase current position
            ratio = current_position / current_distance
            polyline_new.append((1 - ratio) * polyline[current_idx] +
                                ratio * polyline[current_idx + 1])
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
    polyline_new.append(polyline[-1])

    return np.array(polyline_new)


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
    return np.arctan2(direction_vector[1], direction_vector[0])


def sort_lanelet_ids_by_orientation(list_ids_lanelets: List[int], orientation: float, position: np.ndarray,
                                    scenario: Scenario) \
        -> List[int]:
    """Returns the lanelets sorted by relative orientation to the given position and orientation."""

    if len(list_ids_lanelets) <= 1:
        return list_ids_lanelets
    else:
        lanelet_id_list = np.array(list_ids_lanelets)

        def get_lanelet_relative_orientation(lanelet_id):
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            lanelet_orientation = lanelet_orientation_at_position(lanelet, position)
            return np.abs(relative_orientation(lanelet_orientation, orientation))

        orientation_differences = np.array(list(map(get_lanelet_relative_orientation, lanelet_id_list)))
        sorted_indices = np.argsort(orientation_differences)
        return list(lanelet_id_list[sorted_indices])


def sort_lanelet_ids_by_goal(scenario: Scenario, goal: GoalRegion) -> List[int]:
    """Sorts lanelet ids by goal region

    :param goal:
    :param scenario: commonroad scenario
    :return: lanelet id, if the obstacle is out of lanelet boundary (no lanelet is found, therefore return the
    lanelet id of last time step)
    """
    if hasattr(goal, 'lanelets_of_goal_position') and goal.lanelets_of_goal_position is not None:
        goal_lanelet_id_batch_list = list(goal.lanelets_of_goal_position.values())
        goal_lanelet_id_list = [item for sublist in goal_lanelet_id_batch_list for item in sublist]
        goal_lanelet_id_set = set(goal_lanelet_id_list)
        goal_lanelets = [scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id) for goal_lanelet_id in
                         goal_lanelet_id_list]
        goal_lanelets_with_successor = np.array(
            [1.0 if len(set(goal_lanelet.successor).intersection(goal_lanelet_id_set)) > 0 else 0.0 for goal_lanelet
             in goal_lanelets])
        return [x for _, x in sorted(zip(goal_lanelets_with_successor, goal_lanelet_id_list))]

    if goal.state_list is not None and len(goal.state_list) != 0:
        # if len(goal.state_list) > 1:
        #     raise ValueError("More than one goal state is not supported yet!")
        goal_state = goal.state_list[0]

        if hasattr(goal_state, "orientation"):
            goal_orientation: float = (goal_state.orientation.start + goal_state.orientation.end) / 2
        else:
            goal_orientation = 0.0
            warnings.warn("The goal state has no <orientation> attribute! It is set to 0.0")

        if hasattr(goal_state, "position"):
            goal_shape: Shape = goal_state.position
        else:
            goal_shape: Shape = Rectangle(length=0.01, width=0.01)

        # the goal shape has always a shapley object -> because it is a rectangle
        # every shape has a shapely_object but ShapeGroup

        # noinspection PyUnresolvedReferences
        return sort_lanelet_ids_by_orientation(
            scenario.lanelet_network.find_lanelet_by_shape(goal_shape),
            goal_orientation,
            #            np.array(goal_shape.shapely_object.centroid),
            goal_shape.shapely_object.centroid.coords,
            scenario
        )

    raise NotImplementedError("Whole lanelet as goal must be implemented here!")


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """ Computes curvature of the given polyline

    :param polyline: The polyline for the curvature computation
    :return: The curvature of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)
    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    # compute curvature
    curvature = (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))

    return curvature
