import itertools
import sys
import warnings
from enum import Enum
from typing import List, Set, Tuple, Union

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.external.commonroad_route_planner.utility.route import chaikins_corner_cutting, resample_polyline, sort_lanelet_ids_by_goal, sort_lanelet_ids_by_orientation

try:
    import commonroad_dc.pycrccosy as pycrccosy

except ModuleNotFoundError:
    try:
        import pycrccosy

    except ModuleNotFoundError:
        warnings.warn("<route.py> no pycrccosy module found!")


class RouteType(Enum):
    # Survival routes have no specific goal lanelet
    REGULAR = "regular"
    SURVIVAL = "survival"


class Route:
    """Class to represent a route in the scenario."""

    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, list_ids_lanelets: List[int],
                 route_type: RouteType, set_ids_lanelets_permissible: Set[int] = None):
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.lanelet_network = scenario.lanelet_network

        # a route is created given the list of lanelet ids from start to goal
        self.list_ids_lanelets = list_ids_lanelets
        self.type = route_type

        # a section is a list of lanelet ids that are adjacent to a lanelet in the route
        self.list_sections = list()
        self.set_ids_lanelets_in_sections = set()
        self.set_ids_lanelets_opposite_direction = set()

        if set_ids_lanelets_permissible is None:
            self.set_ids_lanelets_permissible = {lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets}
        else:
            self.set_ids_lanelets_permissible = set_ids_lanelets_permissible

        # generate reference path from the list of lanelet ids leading to goal
        self.reference_path = self._generate_reference_path()

        self.path_length = self._compute_path_length_from_polyline(self.reference_path)
        self.path_orientation = self._compute_orientation_from_polyline(self.reference_path)
        self.path_curvature = self._compute_curvature_from_polyline(self.reference_path)

        if 'pycrccosy' in sys.modules:
            # make sure the reference path is already resampled and smoothened before creating a CLCS out of it
            self.clcs = pycrccosy.CurvilinearCoordinateSystem(self.reference_path)

    def retrieve_route_sections(self, is_opposite_direction_allowed: bool = False) -> Union[None, List[List[int]]]:
        """Retrieves route sections for lanelets in the route.

        A section is a list of lanelet ids that are adjacent to a given lanelet.
        """
        if not self.list_sections:
            # compute list of sections
            for id_lanelet in self.list_ids_lanelets:
                # for every lanelet in the route, get its adjacent lanelets
                list_ids_lanelets_in_section = self._get_adjacent_lanelets_ids(id_lanelet,
                                                                               is_opposite_direction_allowed)
                # add lanelet from the route too
                list_ids_lanelets_in_section.append(id_lanelet)
                list_ids_lanelets_in_section.sort()

                if len(self.list_sections) == 0:
                    self.list_sections.append(list_ids_lanelets_in_section)

                elif self.list_sections[-1] != list_ids_lanelets_in_section:
                    # only add new sections if it is not the same as the last section
                    self.list_sections.append(list_ids_lanelets_in_section)

        for section in self.list_sections:
            for id_lanelet in section:
                self.set_ids_lanelets_in_sections.add(id_lanelet)

        return self.list_sections

    def _get_adjacent_lanelets_ids(self, id_lanelet: int, is_opposite_direction_permissible=False) -> list:
        """Recursively gets adj_left and adj_right lanelets of the given lanelet.

        :param id_lanelet: current lanelet id
        :param is_opposite_direction_permissible: specifies if it should give back only the lanelets in the driving
            direction or it should give back the first neighbouring lanelet in the opposite direction
        :return: list of adjacent lanelet ids: all lanelets which are going in the same direction and one-one from the
                 left and right side which are going in the opposite direction, empty lists if there are none
        """
        list_lanelets_adjacent = list()
        lanelet_base = self.lanelet_network.find_lanelet_by_id(id_lanelet)

        # goes in left direction
        lanelet_current = lanelet_base
        id_lanelet_temp = lanelet_current.adj_left
        while id_lanelet_temp is not None:
            # set this lanelet as the current if it goes in the same direction and iterate further
            if id_lanelet_temp in self.set_ids_lanelets_permissible:
                if lanelet_current.adj_left_same_direction:
                    # append the left adjacent lanelet
                    list_lanelets_adjacent.append(id_lanelet_temp)

                    # update lanelet_current
                    lanelet_current = self.lanelet_network.find_lanelet_by_id(id_lanelet_temp)
                    id_lanelet_temp = lanelet_current.adj_left

                else:
                    # if the lanelet is in opposite direction, we add them into a set
                    # if driving in opposite lanelet is allowed, they can be traversed too to form the route
                    self.set_ids_lanelets_opposite_direction.add(id_lanelet_temp)
                    if is_opposite_direction_permissible:
                        list_lanelets_adjacent.append(id_lanelet_temp)
                    break
            else:
                # it is not allowed to drive in that lane, so just break
                break

        # goes in right direction
        lanelet_current = lanelet_base
        id_lanelet_temp = lanelet_current.adj_right
        while id_lanelet_temp is not None:
            # set this lanelet as the current if it goes in the same direction and iterate further
            if id_lanelet_temp in self.set_ids_lanelets_permissible:
                if lanelet_current.adj_right_same_direction:
                    # append the right adjacent lanelet
                    list_lanelets_adjacent.append(id_lanelet_temp)

                    # Update lanelet_current
                    lanelet_current = self.lanelet_network.find_lanelet_by_id(id_lanelet_temp)
                    id_lanelet_temp = lanelet_current.adj_right
                else:
                    # if the lanelet is in opposite direction, we add them into a set
                    # if driving in opposite lanelet is allowed, they can be traversed too to form the route
                    self.set_ids_lanelets_opposite_direction.add(id_lanelet_temp)
                    if is_opposite_direction_permissible:
                        list_lanelets_adjacent.append(id_lanelet_temp)
                    break
            else:
                # it is not allowed to drive in that lane, so just break
                break

        return list_lanelets_adjacent

    def _generate_reference_path(self) -> np.ndarray:
        """Generates a reference path (polyline) out of the given route

        This is done in four steps:
        1. compute lane change instructions
        2. compute the portion of each lanelet based on the instructions
        3. compute the reference path based on the portion
        4. smoothen the reference path

        :return: reference path in 2d numpy array ([[x0, y0], [x1, y1], ...])
        """
        instruction = self._compute_lane_change_instructions()
        list_portions = self._compute_lanelet_portion(instruction)
        reference_path = self._compute_reference_path(list_portions)
        reference_path_smoothed = chaikins_corner_cutting(reference_path)

        return reference_path_smoothed

    def _compute_lane_change_instructions(self) -> List[int]:
        """Computes lane change instruction for planned routes

        The instruction is a list of 0s and 1s, with 0 indicating  no lane change is required
        (driving straight forward0, and 1 indicating that a lane change (to the left or right) is required.
        """
        list_instructions = []
        for idx, id_lanelet in enumerate(self.list_ids_lanelets[:-1]):
            if self.list_ids_lanelets[idx + 1] in self.lanelet_network.find_lanelet_by_id(id_lanelet).successor:
                list_instructions.append(0)
            else:
                list_instructions.append(1)

        # add 0 for the last lanelet
        list_instructions.append(0)

        return list_instructions

    @staticmethod
    def _compute_lanelet_portion(list_instructions: List) -> List[Tuple[float, float]]:
        """Computes the portion of the center vertices of the lanelets required to construct the reference path

        This is done by first grouping the instructions into consecutive sections (each with only 0s or 1s).
        For the group of 0s, as no lane change is required, the whole lanelet is used; for the group of 1s,
        the upper limit of the portion is computed within the group as (idx_lanelet in the group) / (num_lanelets).
        For example, if there are three consecutive lane changes (assuming three lanes are all parallel), the proportion
        would be [0 - 0.25], [0.25 -0.5], [0.5 - 0.75] and [0.75 - 1.0] for these four lanes.
        """

        # returns a list of consecutive instructions
        # e.g. input: [0, 0, 1, 1, 0, 1] output: [[0, 0], [1, 1], [0], [1]]
        list_instructions_consecutive = [list(v) for k, v in itertools.groupby(list_instructions)]

        list_bounds_upper = []
        list_bounds_lower = [0.]
        for instructions in list_instructions_consecutive:
            for idx, instruction in enumerate(instructions):
                if instruction == 0:
                    # goes till the end of the lanelet
                    bound_upper = 1.
                else:
                    # goes only till a specific portion
                    bound_upper = (idx + 1) / (len(instructions) + 1)

                list_bounds_upper.append(bound_upper)

        if len(list_bounds_upper) > 1:
            for idx in range(1, len(list_bounds_upper)):
                if np.isclose(list_bounds_upper[idx - 1], 1.):
                    list_bounds_lower.append(0.)
                else:
                    list_bounds_lower.append(list_bounds_upper[idx - 1])

        assert len(list_bounds_lower) == len(list_bounds_upper) == len(list_instructions), \
            f"The lengths of portions do not match."

        return [(lower, upper) for lower, upper in zip(list_bounds_lower, list_bounds_upper)]

    def _compute_reference_path(self, list_portions, num_vertices_lane_change_max=6,
                                percentage_vertices_lane_change_max=0.1, step_resample=1.0):
        """Computes reference path given the list of portions of each lanelet

        :param list_portions
        :param num_vertices_lane_change_max: number of vertices to perform lane change.
                                             if set to 0, it will produce a zigzagged polyline.
        :param percentage_vertices_lane_change_max: maximum percentage of vertices that should be used for lane change.
        """
        reference_path = None
        num_lanelets_in_route = len(self.list_ids_lanelets)
        for idx, id_lanelet in enumerate(self.list_ids_lanelets):
            lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet)
            # resample the center vertices to prevent too few vertices with too large distances
            vertices_resampled = resample_polyline(lanelet.center_vertices, step_resample)
            num_vertices = len(vertices_resampled)
            num_vertices_lane_change = min(int(num_vertices * percentage_vertices_lane_change_max) + 1,
                                           num_vertices_lane_change_max)
            if reference_path is None:
                idx_start = int(list_portions[idx][0] * num_vertices)
                idx_end = int(list_portions[idx][1] * num_vertices) - num_vertices_lane_change
                # prevent index out of bound
                idx_end = max(idx_end, 1)
                reference_path = vertices_resampled[idx_start:idx_end, :]
            else:
                idx_start = int(list_portions[idx][0] * num_vertices) + num_vertices_lane_change
                # prevent index out of bound
                idx_start = min(idx_start, num_vertices - 1)

                idx_end = int(list_portions[idx][1] * num_vertices)
                # reserve some vertices if it is not the last lanelet
                if idx != (num_lanelets_in_route - 1):
                    idx_end = idx_end - num_vertices_lane_change
                    # prevent index out of bound
                    idx_end = max(idx_end, 1)

                path_to_be_concatenated = vertices_resampled[idx_start:idx_end, :]

                reference_path = np.concatenate((reference_path, path_to_be_concatenated), axis=0)

        reference_path = resample_polyline(reference_path, 2)
        return reference_path

    @staticmethod
    def _compute_path_length_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes the path length of a polyline

        :param polyline: polyline for which path length should be calculated
        :return: path length along polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for path lenth computation p={}'.format(polyline)

        distance = np.zeros((len(polyline),))
        for i in range(1, len(polyline)):
            distance[i] = distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1])

        return np.array(distance)

    @staticmethod
    def _compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes curvature along a polyline

        :param polyline: polyline for which curvature should be calculated
        :return: curvature along  polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)

        x_d = np.gradient(polyline[:, 0])
        x_dd = np.gradient(x_d)
        y_d = np.gradient(polyline[:, 1])
        y_dd = np.gradient(y_d)

        return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))

    @staticmethod
    def _compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes orientation along a polyline

        :param polyline: polyline for which orientation should be calculated
        :return: orientation along polyline
        """
        assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(
            polyline[0, :]) == 2, '<Math>: not a valid polyline. polyline = {}'.format(polyline)
        if len(polyline) < 2:
            raise ValueError('Cannot create orientation from polyline of length < 2')

        orientation = [0]
        for i in range(1, len(polyline)):
            pt1 = polyline[i - 1]
            pt2 = polyline[i]
            tmp = pt2 - pt1
            orientation.append(np.arctan2(tmp[1], tmp[0]))

        return np.array(orientation)

    def orientation(self, position) -> float:
        """
        Calculates orientation of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns orientation of lane at a given position
        """
        return np.interp(position, self.path_length, self.path_orientation)


class RouteCandidateHolder:
    """Class to hold route candidates generated by the route planner"""

    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, list_route_candidates: List[List[int]],
                 route_type: RouteType, set_ids_lanelets_permissible: Set):

        self.scenario = scenario
        self.planning_problem = planning_problem
        self.lanelet_network = self.scenario.lanelet_network

        # create a list of Route objects for all routes found by the route planner which is not empty
        self.list_route_candidates = [Route(scenario, planning_problem, route, route_type, set_ids_lanelets_permissible)
                                      for route in list_route_candidates if route]
        self.num_route_candidates = len(self.list_route_candidates)

        if set_ids_lanelets_permissible is None:
            self.set_ids_lanelets_permissible = {lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets}
        else:
            self.set_ids_lanelets_permissible = set_ids_lanelets_permissible

        self.route_type = route_type

    def retrieve_first_route(self) -> Route:
        return self.list_route_candidates[0]

    def retrieve_best_route_by_orientation(self) -> Union[Route, None]:
        """Retrieves the best route found by some orientation metrics

        If it is the survival scenario, then the first route with idx 0 is returned.
        """
        if not len(self.list_route_candidates):
            return None

        if self.route_type == RouteType.SURVIVAL:
            return self.retrieve_first_route()

        else:
            state_current = self.planning_problem.initial_state
            # sort the lanelets in the scenario based on their orientation difference with the initial state
            list_ids_lanelets_initial_sorted = sort_lanelet_ids_by_orientation(
                self.scenario.lanelet_network.find_lanelet_by_position([state_current.position])[0],
                state_current.orientation,
                state_current.position,
                self.scenario
            )
            # sort the lanelets in the scenario based on the goal region metric
            list_ids_lanelets_goal_sorted = sort_lanelet_ids_by_goal(self.scenario, self.planning_problem.goal)

            list_ids_lanelet_goal_candidates = np.array(
                [route_candidate.list_ids_lanelets[-1] for route_candidate in self.list_route_candidates])

            for id_lanelet_goal in list_ids_lanelets_goal_sorted:
                if id_lanelet_goal in list_ids_lanelet_goal_candidates:
                    list_ids_lanelets_initial_candidates = list()
                    for route_candidate in self.list_route_candidates:
                        if route_candidate.list_ids_lanelets[-1] == id_lanelet_goal:
                            list_ids_lanelets_initial_candidates.append(route_candidate.list_ids_lanelets[0])
                        else:
                            list_ids_lanelets_initial_candidates.append(None)
                    list_ids_lanelets_initial_candidates = np.array(list_ids_lanelets_initial_candidates)

                    for initial_lanelet_id in list_ids_lanelets_initial_sorted:
                        if initial_lanelet_id in list_ids_lanelets_initial_candidates:
                            route = self.list_route_candidates[
                                np.where(list_ids_lanelets_initial_candidates == initial_lanelet_id)[0][0]]
                            return route
            return None

    def retrieve_all_routes(self) -> Tuple[List[Route], int]:
        """ Returns the list of Route objects and the total number of routes"""
        return self.list_route_candidates, self.num_route_candidates

    def __repr__(self):
        return f"{len(self.list_route_candidates)} routeCandidates of scenario {self.scenario.scenario_id}, " \
               f"planning problem {self.planning_problem.planning_problem_id}"

    def __str__(self):
        return self.__repr__()
