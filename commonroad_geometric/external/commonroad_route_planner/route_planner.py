__author__ = "Daniel Tar, Peter Kocsis, Edmond Irani Liu"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0.0"
__maintainer__ = "Edmond Irani Liu"
__email__ = "edmond.irani@tum.de"
__status__ = "Release"

import logging
import os
from datetime import datetime
from enum import Enum
from typing import List, Generator, Set
from pathlib import Path
import networkx as nx
import numpy as np
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.external.commonroad_route_planner.priority_queue import PriorityQueue
from commonroad_geometric.external.commonroad_route_planner.route import RouteType, RouteCandidateHolder
from commonroad_geometric.external.commonroad_route_planner.utility.route import lanelet_orientation_at_position, relative_orientation


class RoutePlanner:
    """Main class for planning routes in CommonRoad scenarios.

    This is a high-level planner that plans on the lanelet level. It returns the best routes for each pair
    of start/goal lanelets, with each route in the form of an ordered list of lanelet IDs. Depending on the
    utilized backend, the best route may have the shortest distance (if using NETWORKX and NETWORKX_REVERSED)
    or may have the lowest cost computed per the heuristic function (if using PRIORITY_QUEUE).
    In survival scenarios (no goal lanelet), the planner advances in the order of forward, right, left when possible.
    """

    class Backend(Enum):
        """Supported backend for constructing the routes.

        NETWORKX: uses built-in functions from the networkx package, tends to change lanes later
        NETWORKX_REVERSED: uses built-in functions from the networkx package, tends to change lanes earlier
        PRIORITY_QUEUE: uses A-star search to find routes, lane changing maneuver depends on the heuristic cost
        """
        NETWORKX = "networkx"
        NETWORKX_REVERSED = "networkx_reversed"
        PRIORITY_QUEUE = "priority_queue"

        @classmethod
        def values(cls):
            return [item.value for item in cls]

    class LaneletNode:
        """Custom node class to represent a lanelet.

        This is used to represent the lanelets as nodes when performing A-star search.
        """

        def __init__(self, id_lanelet: int, lanelet: Lanelet, cost: float, length_current: int):
            self.id = id_lanelet
            self.lanelet = lanelet
            self.cost = cost
            self.count = length_current
            self.parent_node = None

        def __lt__(self, other):
            # define '<' operation for comparison
            return self.cost < other.cost

    class NoSourceLaneletId(Exception):
        def __init__(self):
            self.message = "<RoutePlanner> No initial position given."

    class NoPathFound(Exception):
        def __init__(self, message):
            self.message = message

    def __init__(self, scenario: Scenario,
                 planning_problem: PlanningProblem,
                 set_types_lanelets_forbidden: List[LaneletType] = None,
                 allow_diagonal=False,
                 backend: Backend = Backend.NETWORKX,
                 reach_goal_state: bool = False,
                 log_to_console=False,
                 log_to_file=False):
        """Initialization of a RoutePlanner object.

        :param scenario: scenario on which the routes should be planned
        :param planning_problem: planning problem for which the routes should be planned
        :param set_types_lanelets_forbidden: set of lanelet types which should be avoided during route planning
        :param allow_diagonal: indicates whether diagonal movements are allowed - experimental
        :param backend: the backend to be used
        :param reach_goal_state: indicates whether the reference path should pass through the goal state (position).
        :param log_to_console: indicates whether the outputs should be logged to the console
        :param log_to_file: indicates whether the outputs should be logged to file
        """
        # setting backend
        if backend not in RoutePlanner.Backend.values():
            backend = RoutePlanner.Backend.NETWORKX
        self.backend = backend

        self.scenario = scenario
        self.scenario_id = scenario.scenario_id
        self.lanelet_network = scenario.lanelet_network
        self.planning_problem = planning_problem

        if set_types_lanelets_forbidden is None:
            set_types_lanelets_forbidden = set()
        self.set_types_lanelets_forbidden = set_types_lanelets_forbidden

        self.allow_diagonal = allow_diagonal
        self.reach_goal_state = reach_goal_state

        self.logger = logging.getLogger(f"Route Planner [{self.scenario_id}]")
        self._init_logger(log_to_console=log_to_console, log_to_file=log_to_file)

        # find permissible lanelets
        list_lanelets_filtered = self._filter_lanelets_by_type(self.lanelet_network.lanelets,
                                                               self.set_types_lanelets_forbidden)
        self.set_ids_lanelets_permissible = {lanelet_permissible.lanelet_id
                                             for lanelet_permissible in list_lanelets_filtered}

        # examine initial and goal lanelet ids
        self.id_lanelets_start = self._retrieve_ids_lanelets_start()

        self.ids_lanelets_goal, self.ids_lanelets_goal_original = self._retrieve_ids_lanelets_goal()

        if self.reach_goal_state and not self.ids_lanelets_goal:
            # if the predecessors of the goal states cannot be reached, fall back to reaching the goal lanelets
            self.reach_goal_state = False
            self.ids_lanelets_goal, self.ids_lanelets_goal_original = self._retrieve_ids_lanelets_goal()

        self._create_lanelet_network_graph()

    def _init_logger(self, log_to_console=True, log_to_file=True, add_timestamp_to_log_file=True):
        """Initializes a logger."""
        # path relative to the running script
        log_file_dir = Path("solutions/logs/scenario_logs")
        log_file_name = Path("route_planner_result_with_priority_queue_backend")

        # release_logger(self.logger)
        self.logger.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s\t\t%(name)s\t%(levelname)s\t%(message)s')
        file_formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')

        if log_to_console:
            # create console handler
            console_handler = logging.StreamHandler()
            # set the level of logging to console
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        if log_to_file:
            date_time_string = ''
            if add_timestamp_to_log_file:
                now = datetime.now()  # current date and time
                date_time_string = now.strftime("_%Y_%m_%d_%H-%M-%S")

            # if directory not exists create it
            os.makedirs(log_file_dir, exist_ok=True)

            log_file_path = os.path.join(log_file_dir,
                                         "{}_{}{}.log".format(self.scenario_id, log_file_name, date_time_string))
            file_handler = logging.FileHandler(log_file_path)
            # set the level of logging to file
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.debug("Using backend: {}".format(self.backend))

    def _retrieve_ids_lanelets_start(self):
        """Retrieves the ids of the lanelets in which the initial position is situated"""
        if hasattr(self.planning_problem.initial_state, 'position'):
            post_start = self.planning_problem.initial_state.position
            # noinspection PyTypeChecker
            list_ids_lanelets_start = self.lanelet_network.find_lanelet_by_position([post_start])[0]

            list_ids_lanelets_start = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_start))

            # Check if any of the start positions are during an overtake:
            # if the car is not driving in the correct direction for the lanelet,
            # it will also consider routes taking an adjacent lanelet in the opposite direction
            self.ids_lanelets_start_overtake = list()
            if (hasattr(self.planning_problem.initial_state, 'orientation')
                    and not self.planning_problem.initial_state.is_uncertain_orientation):
                orientation = self.planning_problem.initial_state.orientation

                for id_lanelet_start in list_ids_lanelets_start:
                    lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet_start)
                    lanelet_angle = lanelet_orientation_at_position(lanelet, post_start)

                    # Check if the angle difference is larger than 90 degrees
                    if abs(relative_orientation(orientation, lanelet_angle)) > 0.5 * np.pi:
                        if (lanelet.adj_left is not None and not lanelet.adj_left_same_direction
                                and lanelet.adj_left in self.set_ids_lanelets_permissible):
                            self.ids_lanelets_start_overtake.append((id_lanelet_start, lanelet.adj_left))

                        elif (lanelet.adj_right is not None and not lanelet.adj_right_same_direction
                              and lanelet.adj_right in self.set_ids_lanelets_permissible):
                            self.ids_lanelets_start_overtake.append((id_lanelet_start, lanelet.adj_right))

            if len(list_ids_lanelets_start) > 1:
                self.logger.info("Multiple start lanelet IDs: some may fail to reach goal lanelet")
        else:
            self.logger.critical("No initial position in the given planning problem")
            raise self.NoSourceLaneletId()

        return list_ids_lanelets_start

    def _retrieve_ids_lanelets_goal(self):
        """Retrieves the ids of the lanelets in which the goal position is situated"""
        list_ids_lanelets_goal = list()
        list_ids_lanelets_goal_original = list()

        if hasattr(self.planning_problem.goal, 'lanelets_of_goal_position'):
            if self.planning_problem.goal.lanelets_of_goal_position is None:
                self.logger.debug("No goal lanelet given")
            else:
                self.logger.debug("Goal lanelet given")
                # the goals are stored in a dict, one goal can consist of multiple lanelets
                # now we just iterate over the goals and add every ID which we find to
                # the goal_lanelet_ids list
                for list_ids_lanelets_pos_goal in list(self.planning_problem.goal.lanelets_of_goal_position.values()):
                    list_ids_lanelets_goal.extend(list_ids_lanelets_pos_goal)

                list_ids_lanelets_goal = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_goal))

        if list_ids_lanelets_goal:
            self.reach_goal_state = False

        elif hasattr(self.planning_problem.goal, 'state_list'):
            for idx, state in enumerate(self.planning_problem.goal.state_list):
                if hasattr(state, 'position'):
                    if hasattr(state.position, 'center'):
                        pos_goal = state.position.center

                    else:
                        pos_goal = state.position
                    [list_ids_lanelets_pos_goal] = self.lanelet_network.find_lanelet_by_position([pos_goal])
                    list_ids_lanelets_pos_goal = list(self._filter_allowed_lanelet_ids(list_ids_lanelets_pos_goal))

                    if self.reach_goal_state:
                        # we want to reach the goal states (not just the goal lanelets), here we instead demand
                        # reaching the predecessor lanelets of the goal states
                        list_ids_lanelets_goal_original = list_ids_lanelets_pos_goal.copy()
                        list_ids_lanelets_pos_goal.clear()

                        for id_lanelet_goal in list_ids_lanelets_goal_original:
                            lanelet_goal = self.lanelet_network.find_lanelet_by_id(id_lanelet_goal)
                            # make predecessor as goal
                            list_ids_lanelets_pos_goal.extend(lanelet_goal.predecessor)

                    if list_ids_lanelets_pos_goal:
                        list_ids_lanelets_goal.extend(list_ids_lanelets_pos_goal)
                        self.logger.debug("Goal lanelet IDs estimated from goal shape in state [{}]".format(idx))
                    else:
                        self.logger.debug(
                            "No Goal lanelet IDs could be determined from the goal shape in state [{}]".format(idx))

        # remove duplicates and reset to none if no lanelet IDs found
        if list_ids_lanelets_goal:
            # remove duplicates and sort in ascending order
            list_ids_lanelets_goal = sorted(list(dict.fromkeys(list_ids_lanelets_goal)))
        else:
            list_ids_lanelets_goal = None

        return list_ids_lanelets_goal, list_ids_lanelets_goal_original

    def _create_lanelet_network_graph(self):
        """Creates a directed graph of lanelets."""

        if self.ids_lanelets_goal is None:
            # if there is no goal lanelet ids then it is a survival scenario and
            # we do not need to make a graph from the lanelet network
            self.route_type = RouteType.SURVIVAL
            self.logger.info("Survival Scenario: No lanelet graph created.")

        else:
            # construct directed graph
            self.route_type = RouteType.REGULAR

            if self.backend == RoutePlanner.Backend.NETWORKX:
                if self.allow_diagonal:
                    self.logger.warning("Diagonal node connection not tested")
                    self.digraph = self._create_graph_from_lanelet_network_diagonal()

                else:
                    self.digraph = self._create_graph_from_lanelet_network()

            elif self.backend == RoutePlanner.Backend.NETWORKX_REVERSED:
                if self.allow_diagonal:
                    self.logger.warning("Diagonal node connection not implemented")
                    self.digraph = self._create_reversed_graph_from_lanelet_network_lane_change()

                else:
                    self.digraph = self._create_reversed_graph_from_lanelet_network()

            elif self.backend == RoutePlanner.Backend.PRIORITY_QUEUE:
                if self.allow_diagonal:
                    self.logger.critical("diagonal search with custom backend is not implemented")

    def plan_routes(self):
        """Plans routes for every pair of start/goal lanelets.

        If no goal lanelet ID is given then return a survival route.
        :return: list of lanelet ids from start to goal.
        """
        self.logger.info("Route planner started")
        # route is a list that holds lists of lanelet ids from start lanelet to goal lanelet
        list_routes = list()

        # iterate through start lanelet ids
        for id_lanelet_start in self.id_lanelets_start:
            if self.ids_lanelets_goal:
                # iterate through goal lanelet ids
                for id_lanelet_goal in self.ids_lanelets_goal:
                    list_lists_ids_lanelets = list()
                    if self.backend == RoutePlanner.Backend.NETWORKX:
                        list_lists_ids_lanelets = self._find_routes_networkx(id_lanelet_start, id_lanelet_goal)

                    elif self.backend == RoutePlanner.Backend.NETWORKX_REVERSED:
                        list_lists_ids_lanelets = self._find_routes_networkx_reversed(id_lanelet_start, id_lanelet_goal)

                    elif self.backend == RoutePlanner.Backend.PRIORITY_QUEUE:
                        list_lists_ids_lanelets = self._find_routes_priority_queue(id_lanelet_start, id_lanelet_goal)

                    if list_lists_ids_lanelets:
                        if self.reach_goal_state:
                            # append the original goal lanelet back to the found route
                            for id_lanelet_goal_original in self.ids_lanelets_goal_original:
                                for list_ids_lanelets in list_lists_ids_lanelets:
                                    list_routes.append(list_ids_lanelets + [id_lanelet_goal_original])

                        else:
                            list_routes.extend(list_lists_ids_lanelets)

            else:
                # no goal lanelet, find survival route
                list_lists_ids_lanelets = self._find_survival_route(id_lanelet_start)
                list_routes.append(list_lists_ids_lanelets)

        return RouteCandidateHolder(self.scenario, self.planning_problem, list_routes,
                                    self.route_type, self.set_ids_lanelets_permissible)

    @staticmethod
    def _filter_lanelets_by_type(list_lanelets_to_filter: List[Lanelet],
                                 set_types_lanelets_forbidden: Set[LaneletType]) -> Generator[Lanelet, None, None]:
        """Filters lanelets with the set of forbidden types.

        :param list_lanelets_to_filter: The list of the lanelets which should be filtered
        :return: List of desirable lanelets
        """
        for lanelet in list_lanelets_to_filter:
            if len(lanelet.lanelet_type.intersection(set_types_lanelets_forbidden)) == 0:
                yield lanelet

    def _filter_allowed_lanelet_ids(self, list_ids_lanelets_to_filter: List[int]) \
            -> Generator[int, None, None]:
        """Filters lanelets with the list of ids of forbidden lanelets.

        :param list_ids_lanelets_to_filter: The list of the lanelet ids which should be filtered
        :return: List of desirable lanelets
        """
        for id_lanelet in list_ids_lanelets_to_filter:
            if id_lanelet in self.set_ids_lanelets_permissible:
                yield id_lanelet

    def _find_survival_route(self, id_lanelet_start: int) -> List:
        """Finds a route along the lanelet network for survival scenarios.

        The planner advances in the order of forward, right, left whenever possible.
        Notes:
            - it only considers lanes with same driving direction
            - the priorities of right and left should be swapped for left-hand traffic countries, e.g. UK
            - it goes until the end of the lanelet network or when it is hits itself (like dying in the Snake game)

        :param id_lanelet_start: the initial lanelet where we start from
        :return: route that consists of a list of lanelet IDs
        """
        route = list()
        id_lanelet_current = id_lanelet_start
        lanelet = self.lanelet_network.find_lanelet_by_id(id_lanelet_current)

        while id_lanelet_current not in route:
            route.append(lanelet.lanelet_id)

            found_new_lanelet = False
            if lanelet.successor:
                # naively select the first successors
                lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.successor[0])
                found_new_lanelet = True

            if not found_new_lanelet and lanelet.adj_right and lanelet.adj_right_same_direction:
                # try to go right
                lanelet_adj_right = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                if len(lanelet_adj_right.successor):
                    # right lanelet has successor
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                    found_new_lanelet = True

            if not found_new_lanelet and lanelet.adj_left and lanelet.adj_left_same_direction:
                # try to go left
                lanelet_adj_left = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                if len(lanelet_adj_left.successor):
                    # left lanelet has successor
                    lanelet = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                    found_new_lanelet = True

            if not found_new_lanelet:
                # no possible route to advance
                break
            else:
                # set lanelet
                id_lanelet_current = lanelet.lanelet_id

        return route

    def _create_reversed_graph_from_lanelet_network(self) -> nx.DiGraph:
        """Builds a graph from the lanelet network.

        Edges are added from the predecessor relations between lanelets.
        :return: created graph from lanelet network
        """
        graph = nx.DiGraph()
        nodes = list()
        edges = list()
        for lanelet in self.lanelet_network.lanelets:
            if lanelet.lanelet_id not in self.set_ids_lanelets_permissible:
                continue

            nodes.append(lanelet.lanelet_id)

            # add edge if id_predecessor lanelets exist
            for id_predecessor in lanelet.predecessor:
                if id_predecessor not in self.set_ids_lanelets_permissible:
                    continue
                edges.append((lanelet.lanelet_id, id_predecessor, {'weight': lanelet.distance[-1]}))

            # add edge if left lanelet
            id_adj_left = lanelet.adj_left
            if id_adj_left and lanelet.adj_left_same_direction and id_adj_left in self.set_ids_lanelets_permissible:
                edges.append((lanelet.lanelet_id, id_adj_left, {'weight': 4.0}))

            # add edge if right lanelet
            id_adj_right = lanelet.adj_right
            if id_adj_right and lanelet.adj_right_same_direction and id_adj_right in self.set_ids_lanelets_permissible:
                edges.append((lanelet.lanelet_id, id_adj_right, {'weight': 4.0}))

        # Edges in case of overtake during starting state
        for id_start, id_adj in self.ids_lanelets_start_overtake:
            edges.append((id_adj, id_start, {'weight': 1.0}))

        # add all nodes and edges to the graph
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        return graph

    def _create_graph_from_lanelet_network_diagonal(self) -> nx.DiGraph:
        """Builds a graph from the lanelet network allowing diagonal lane changes

        :return: created graph from lanelet network with diagonal lane changes
        """
        graph = nx.DiGraph()
        nodes = list()
        edges = list()

        # check for each lanelet for diagonal successors
        for lanelet in self.lanelet_network.lanelets:
            nodes.append(lanelet.lanelet_id)

            # check if lanelet has id_successor
            # add edge if succeeding lanelet
            for id_successor in lanelet.successor:
                successor_lanelet = self.lanelet_network.find_lanelet_by_id(id_successor)
                edges.append((lanelet.lanelet_id, successor_lanelet.lanelet_id, {'weight': lanelet.distance[-1]}))

                # check for diagonal left succeeding lanelet
                if successor_lanelet.adj_left and lanelet.adj_left_same_direction:
                    edges.append((lanelet.lanelet_id, successor_lanelet.adj_left, {'weight': 0}))

                # check for diagonal right succeeding lanelet
                if successor_lanelet.adj_right and lanelet.adj_right_same_direction:
                    edges.append((lanelet.lanelet_id, successor_lanelet.adj_right, {'weight': 0}))

            # check if succeeding lanelet of right lanelet (e.g. turning lane highway)
            if lanelet.adj_right and lanelet.adj_right_same_direction:
                l_right = self.lanelet_network.find_lanelet_by_id(lanelet.adj_right)

                # check for diagonal right succeeding lanelet
                for right_successor in l_right.successor:

                    # if not already in graph add it
                    if (lanelet.lanelet_id, right_successor, {'weight': 0}) not in edges:
                        edges.append((lanelet.lanelet_id, right_successor, {'weight': 0}))

            # check if succeeding lanelet of right lanelet (e.g. turning lane highway)
            if lanelet.adj_left and lanelet.adj_left_same_direction:
                l_left = self.lanelet_network.find_lanelet_by_id(lanelet.adj_left)

                # check for diagonal left succeeding lanelet
                for left_successor in l_left.successor:

                    # if not already in graph add it
                    if (lanelet.lanelet_id, left_successor, {'weight': 0}) not in edges:
                        edges.append((lanelet.lanelet_id, left_successor, {'weight': 0}))

        # Edges in case of overtake during starting state
        for id_start, id_adj in self.ids_lanelets_start_overtake:
            edges.append((id_start, id_adj, {'weight': 0}))

        # add all nodes and edges to graph
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _create_graph_from_lanelet_network(self) -> nx.DiGraph:
        """Builds a graph from the lanelet network

        Edges are added from the successor relations between lanelets.

        :return: created graph from lanelet network
        """

        graph = nx.DiGraph()
        nodes = list()
        edges = list()

        for lanelet in self.lanelet_network.lanelets:
            # only accept allowed lanelets
            if lanelet.lanelet_id not in self.set_ids_lanelets_permissible:
                continue

            nodes.append(lanelet.lanelet_id)

            # add edge if succeeding lanelets exist
            for id_successor in lanelet.successor:
                if id_successor not in self.set_ids_lanelets_permissible:
                    continue
                edges.append((lanelet.lanelet_id, id_successor, {'weight': lanelet.distance[-1]}))

            # add edge if left lanelet
            id_adj_left = lanelet.adj_left
            if id_adj_left and lanelet.adj_left_same_direction and id_adj_left in self.set_ids_lanelets_permissible:
                edges.append((lanelet.lanelet_id, id_adj_left, {'weight': 4.0}))

            # add edge if right lanelet
            id_adj_right = lanelet.adj_right
            if id_adj_right and lanelet.adj_right_same_direction and id_adj_right in self.set_ids_lanelets_permissible:
                edges.append((lanelet.lanelet_id, id_adj_right, {'weight': 4.0}))

        # Edges in case of overtake during starting state
        for id_start, id_adj in self.ids_lanelets_start_overtake:
            edges.append((id_start, id_adj, {'weight': 1.0}))

        # add all nodes and edges to graph
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def _create_reversed_graph_from_lanelet_network_lane_change(self):
        raise NotImplementedError

    def _find_routes_networkx(self, id_lanelet_start: int, id_lanelet_goal: int = None) -> List[List]:
        """Find all shortest paths using networkx module

        This tends to change lane late.
        :param id_lanelet_start: ID of start lanelet
        :param id_lanelet_goal: ID of goal lanelet
        :return: list of lists of lanelet IDs
        """
        list_lanelets = list()

        if id_lanelet_start is None:
            raise self.NoSourceLaneletId()

        if id_lanelet_goal is None:
            self.logger.info("Survival scenario")
            return self._find_survival_route(id_lanelet_start)

        try:
            list_lanelets = list(nx.all_shortest_paths(self.digraph, source=id_lanelet_start, target=id_lanelet_goal,
                                                       weight='weight', method='dijkstra'))
        except nx.exception.NetworkXNoPath:
            # it is a normal behaviour because of the overlapping lanelets in a road network
            self.logger.debug(f"""The goal lanelet with ID [{id_lanelet_goal}] cannot be reached
                              from the start lanelet with ID [{id_lanelet_start}]""")
        return list_lanelets

    def _find_routes_networkx_reversed(self, id_lanelet_start: int, id_lanelet_goal: int = None) -> List[List]:
        """Find all shortest paths (reversed) using the networkx module

        This tends to change lane early.

        :param id_lanelet_start: ID of source lanelet
        :param id_lanelet_goal: ID of goal lanelet
        :return: list of lists of lanelet IDs
        """
        list_lanelets = list()

        if id_lanelet_start is None:
            raise self.NoSourceLaneletId()

        if id_lanelet_goal is None:
            self.logger.info("Survival scenario")
            return self._find_survival_route(id_lanelet_start)

        try:
            list_lanelets.append(list(nx.shortest_path(self.digraph, source=id_lanelet_goal, target=id_lanelet_start,
                                                       weight='weight', method='dijkstra'))[::-1])
        except nx.exception.NetworkXNoPath:
            # it is a normal behaviour because of the overlapping lanelets in a road network
            self.logger.debug(f"""The goal lanelet with ID [{id_lanelet_goal}] cannot be reached
                                          from the start lanelet with ID [{id_lanelet_start}]""")
        return list_lanelets

    def _find_routes_priority_queue(self, id_lanelet_start: int, id_lanelet_goal: int = None) -> List[List]:
        """Find the shortest paths using own implementation (A-star)

        Lane change depends on the heuristic cost.

        :param id_lanelet_start: ID of source lanelet
        :param id_lanelet_goal: ID of goal lanelet
        :return: list of lists of lanelet IDs
        """
        list_lanelets = list()

        if id_lanelet_start is None:
            raise self.NoSourceLaneletId()

        if id_lanelet_goal is None:
            self.logger.debug("Survival scenario")
            list_lanelets.append(self._find_survival_route(id_lanelet_start))
        else:
            try:
                list_lanelets.append(self._find_route_astar(id_lanelet_start, id_lanelet_goal))
            except self.NoPathFound:
                # it is a normal behaviour because of the overlapping lanelets in a road network
                self.logger.debug(f"""The goal lanelet with ID [{id_lanelet_goal}] cannot be reached
                                                          from the start lanelet with ID [{id_lanelet_start}]""")
        return list_lanelets

    @staticmethod
    def _calc_cost_travel(current: Lanelet) -> float:
        # use the length of the lanelet as the travel cost
        return current.distance[-1]

    @staticmethod
    def _calc_cost_heuristic(lanelet_current: Lanelet, lanelet_goal: Lanelet) -> float:
        # use the distance between the vertices of the center lines as the heuristic cost
        diff = lanelet_goal.center_vertices[0] - lanelet_current.center_vertices[-1]
        return np.sqrt(np.dot(diff, diff))

    def _add_child(self, parent_node: LaneletNode, next_lanelet_id: int, target: Lanelet, extra_cost: float = 0.0):
        next_lanelet = self.lanelet_network.find_lanelet_by_id(next_lanelet_id)

        frontier_lanelet_ids = self.frontier.get_item_ids()
        cost = extra_cost + parent_node.cost + self._calc_cost_travel(next_lanelet) + self._calc_cost_heuristic(
            next_lanelet,
            target)

        node = RoutePlanner.LaneletNode(next_lanelet_id, next_lanelet, cost, parent_node.count + 1)
        node.parent_node = parent_node

        if (next_lanelet_id not in self.explored) and (next_lanelet_id not in frontier_lanelet_ids):
            self.frontier.push(next_lanelet_id, node, cost)
        elif next_lanelet_id in frontier_lanelet_ids:
            self.frontier.update_item_if_exists(next_lanelet_id, node, cost)

    def _find_route_astar(self, id_lanelet_start, id_lanelet_goal: int) -> List:

        if id_lanelet_start not in self.set_ids_lanelets_permissible:
            raise self.NoPathFound("Start lanelet is not in the permissible set. Check its lanelet type.")

        if id_lanelet_goal not in self.set_ids_lanelets_permissible:
            raise self.NoPathFound("Goal lanelet is not in the permissible set. Check its lanelet type.")

        self.frontier = PriorityQueue()
        self.explored = set()

        lanelet_current = self.lanelet_network.find_lanelet_by_id(id_lanelet_start)
        lanelet_goal = self.lanelet_network.find_lanelet_by_id(id_lanelet_goal)

        # in astar search, the cost of a node f is the traveled cost g + heuristic cost h
        cost_travel = self._calc_cost_travel(lanelet_current)
        cost_heuristic = self._calc_cost_heuristic(lanelet_current, lanelet_goal)
        node = RoutePlanner.LaneletNode(id_lanelet_start, lanelet_current, cost_travel + cost_heuristic, 1)
        self.frontier.push(node.id, node, node.cost)

        # execute the search
        while not self.frontier.is_empty():
            node: RoutePlanner.LaneletNode = self.frontier.pop()

            if node is None:
                # maybe the frontier is not empty but only contains invalid elements
                continue
            elif node.id == id_lanelet_goal:
                # reached goal
                break
            # add node to explored set (close list)
            self.explored.add(node.id)

            # expand node
            lanelet_current = node.lanelet
            # add successors
            for id_successor in lanelet_current.successor:
                if id_successor in self.set_ids_lanelets_permissible:
                    self._add_child(node, id_successor, lanelet_goal, node.cost)

            # if we are changing lanelets then remove the lanelet lengths because it would be added twice
            lanelet_length = self._calc_cost_travel(lanelet_current)

            # add left lanelet
            adj_left_id = lanelet_current.adj_left
            if lanelet_current.adj_left_same_direction and adj_left_id and adj_left_id in self.set_ids_lanelets_permissible:
                self._add_child(node, adj_left_id, lanelet_goal, 1.0 - lanelet_length)

                if self.allow_diagonal:
                    left_lanelet_successor_ids = self.lanelet_network.find_lanelet_by_id(adj_left_id).successor
                    for left_lanelet_successor_id in left_lanelet_successor_ids:
                        self._add_child(node, left_lanelet_successor_id, lanelet_goal, 0.9)

            # add right lanelet
            adj_right_id = lanelet_current.adj_right
            if lanelet_current.adj_right_same_direction and adj_right_id and adj_right_id in self.set_ids_lanelets_permissible:
                self._add_child(node, adj_right_id, lanelet_goal, 1.0 - lanelet_length)

                if self.allow_diagonal:
                    right_lanelet_successor_ids = self.lanelet_network.find_lanelet_by_id(adj_right_id).successor
                    for right_lanelet_successor_id in right_lanelet_successor_ids:
                        self._add_child(node, right_lanelet_successor_id, lanelet_goal, 0.9)
        else:
            raise self.NoPathFound(
                "The Target lanelet_id [{}] cannot be reached from Source [{}]".format(id_lanelet_goal,
                                                                                       id_lanelet_start))
        list_ids_lanelets_reversed = list()
        # add ids by looking up to parent node
        while node:
            list_ids_lanelets_reversed.append(node.id)
            node = node.parent_node

        # reverse the list to obtain the correct order from start to goal
        return list_ids_lanelets_reversed[::-1]
