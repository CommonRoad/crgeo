from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Set, TYPE_CHECKING, Tuple, Union

import numpy as np
from commonroad.geometry.shape import Rectangle, Shape, ShapeGroup
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.geometry import ContinuousPolyline
from commonroad_geometric.common.io_extensions.lanelet_network import find_lanelet_by_id
from commonroad_geometric.external.commonroad_rl.navigator import Navigator
from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import RespawnerSetupFailure

if TYPE_CHECKING:
    from commonroad_geometric.simulation.base_simulation import BaseSimulation


logger = logging.getLogger(__name__)


EGO_ROUTE_POLYLINE_RESOLUTION = 1000


class EgoRoute(AutoReprMixin):
    """
    High-level interface for route/trajectory planning problem of EgoVehicle in EgoVehicleSimulation.
    Abstracts away PlanningProblemSet whether loaded from scenario file or created randomly.
    Guaranteeing that there is always a PlanningProblem and Navigator available.
    """

    def __init__(
        self,
        simulation: BaseSimulation,
        planning_problem_set: PlanningProblemSet
    ) -> None:
        self._simulation: BaseSimulation = simulation
        self._scenario: Scenario = simulation.initial_scenario
        self._planning_problem_set: PlanningProblemSet
        self._planning_problem: PlanningProblem
        self._goal_region: GoalRegion
        self._goal_lanelet: Optional[Lanelet] = None
        self._lanelet_id_route: List[int]
        self._lanelet_id_route_set: Set[int]
        self._navigator: Optional[Navigator] = None
        self._planned_ego_polyline: Optional[ContinuousPolyline] = None
        self._extended_path_polyline: Optional[ContinuousPolyline] = None
        self._look_ahead_point: Optional[np.ndarray] = None
        self._look_ahead_distance: Optional[float] = None
        self.planning_problem_set = planning_problem_set

    @property
    def look_ahead_distance(self) -> Optional[float]:
        return self._look_ahead_distance

    @look_ahead_distance.setter
    def look_ahead_distance(self, value: float) -> None:
        self._look_ahead_distance = value

    @property
    def look_ahead_point(self) -> Optional[np.ndarray]:
        return self._look_ahead_point

    def set_look_ahead_point(self, x: Union[float, np.ndarray]) -> Tuple[float, np.ndarray]:
        assert self.look_ahead_distance is not None
        path = self.planning_problem_path_polyline
        assert path is not None

        if isinstance(x, float):
            arclength = x
        else:
            arclength = path.get_projected_arclength(x)

        look_ahead_arclength = min(path.length, arclength + self.look_ahead_distance)
        look_ahead_point = path(look_ahead_arclength)

        self._look_ahead_point = look_ahead_point

        return look_ahead_arclength, look_ahead_point

    @property
    def planning_problem_set(self) -> PlanningProblemSet:
        return self._planning_problem_set

    @planning_problem_set.setter
    def planning_problem_set(self, value: PlanningProblemSet):
        self._planning_problem_set = value
        self._navigator = None
        self._extended_path_polyline = None
        if self._planning_problem_set is None or len(self.planning_problem_set.planning_problem_dict) == 0:
            self._planning_problem = None
            self._lanelet_id_route = []
            self._lanelet_id_route_set = set()
            logger.debug(f"Set empty ego route")
            return

        self._planning_problem = next(iter(self.planning_problem_set.planning_problem_dict.values()))
        gs = []
        for state in self._planning_problem.goal.state_list:
            if isinstance(state.position, ShapeGroup):
                state.position = state.position.shapes[0]
                gs.append(state)
                break
            elif isinstance(state.position, (Rectangle, Shape)):
                gs.append(state)
                break
        assert len(gs) == 1
        self._planning_problem.goal.state_list = gs
        self._planning_problem.goal._lanelets_of_goal_position = None
        # self._planning_problem.goal.state_list = [gs[0].position if isinstance(gs[0], State) else gs[0]]

        try:
            self.planning_problem_path = self.navigator.route.reference_path
        except Exception as e:
            raise RespawnerSetupFailure from e

        self.lanelet_id_route = self.navigator.route.list_ids_lanelets

        logger.debug(f"Set ego route: {self.lanelet_id_route}")
        # start_lanelet_ids = self._scenario.lanelet_network.find_lanelet_by_position([self._planning_problem.initial_state.position])[0]
        # if isinstance(self._planning_problem.goal.state_list[0].position, ShapeGroup):
        #     goal_shape = self._planning_problem.goal.state_list[0].position.shapes[0]
        # else:
        #     goal_shape = self._planning_problem.goal.state_list[0].position
        # goal_lanelet_ids = self._scenario.lanelet_network.find_lanelet_by_position([goal_shape.center])[0]
        # for start_lanelet_id in start_lanelet_ids:
        #     for goal_lanelet_id in goal_lanelet_ids:
        #         if start_lanelet_id == goal_lanelet_id:
        #             self.lanelet_id_route = [start_lanelet_id]
        #             return
        #         if start_lanelet_id in self._simulation.routes and goal_lanelet_id in self._simulation.routes[start_lanelet_id]:
        #             self.lanelet_id_route = self._simulation.routes[start_lanelet_id][goal_lanelet_id]
        #             return
        #         # Looking for partial routes
        #         for goal_id_to_route in self._simulation.routes.values():
        #             for route in goal_id_to_route.values():
        #                 if start_lanelet_id in route and goal_lanelet_id in route:
        #                     self.lanelet_id_route = route[route.index(start_lanelet_id):route.index(goal_lanelet_id) + 1]
        #                     return

        # raise EgoVehicleRespawnerSetupFailure(f"EgoRoute failed to assign lanelet route for {self._scenario} with planning problem {self._planning_problem}")

    @property
    def planning_problem(self) -> PlanningProblem:
        return self._planning_problem

    @property
    def goal_region(self) -> GoalRegion:
        return self._planning_problem.goal

    @property
    def lanelet_id_route(self) -> List[int]:
        return self._lanelet_id_route

    @lanelet_id_route.setter
    def lanelet_id_route(self, value: List[int]) -> None:
        self._lanelet_id_route = value
        self._lanelet_id_route_set = set(value)
        assert len(value) > 0

    @property
    def lanelet_id_route_set(self) -> Set[int]:
        return self._lanelet_id_route_set

    def locate_route_position(self, lanelet_id: int) -> Tuple[Optional[int], Optional[int]]:
        current_idx, current_lid = None, None
        for idx, lid in enumerate(self.lanelet_id_route):
            if lanelet_id == lid:
                current_idx, current_lid = idx, lid
                break
        if current_idx is not None and current_idx < len(self._lanelet_id_route) - 1:
            next_lid = self.lanelet_id_route[current_idx + 1]
        else:
            next_lid = None
        return current_lid, next_lid

    @property
    def goal_lanelet(self) -> Lanelet:
        if self._goal_lanelet is None:
            if self.goal_region.lanelets_of_goal_position is not None:
                goal_lanelet_id = self.goal_region.lanelets_of_goal_position[0][0]
            else:
                goal_lanelet_id = self._scenario.lanelet_network.find_lanelet_by_position([self.goal_region.state_list[0].position.center])[0][0]
            goal_lanelet = find_lanelet_by_id(self._scenario.lanelet_network, goal_lanelet_id)
            self._goal_lanelet = goal_lanelet
        return self._goal_lanelet

    @property
    def planning_problem_path(self) -> Optional[np.ndarray]:
        if self.planning_problem_path_polyline is None:
            return None
        return self.planning_problem_path_polyline.waypoints

    @planning_problem_path.setter
    def planning_problem_path(self, trajectory: Optional[np.ndarray]) -> None:
        if trajectory is None:
            self._planned_ego_polyline = None
        else:
            self._planned_ego_polyline = EgoRoute.create_polyline_from_waypoints(trajectory)

    @property
    def planning_problem_path_polyline(self) -> Optional[ContinuousPolyline]:
        return self._planned_ego_polyline

    @property
    def extended_path_polyline(self) -> Optional[ContinuousPolyline]:
        if self._extended_path_polyline is None:
            waypoints = []
            for i, lid in enumerate(self.lanelet_id_route):
                waypoints_i = self._simulation.get_lanelet_center_polyline(lid).waypoints
                if i == 0:
                    waypoints.append(waypoints_i)
                else:
                    waypoints.append(waypoints_i[1:])
            self._extended_path_polyline = ContinuousPolyline.merge(*waypoints)
        return self._extended_path_polyline

    # @planning_problem_path_polyline.setter
    # def planning_problem_path_polyline(self, polyline: ContinuousPolyline) -> None:
    #     self._planned_ego_polyline = polyline

    @staticmethod
    def create_polyline_from_waypoints(waypoints: np.ndarray, /) -> ContinuousPolyline:
        return ContinuousPolyline(waypoints=waypoints, waypoint_resolution=EGO_ROUTE_POLYLINE_RESOLUTION)

    @property
    def navigator(self) -> Navigator:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._navigator is None:
                self._navigator = Navigator(
                    scenario=self._scenario,
                    planning_problem=self.planning_problem
                )
        return self._navigator

    def check_if_has_reached_goal(self, ego_state: State) -> bool:
        has_reached_goal = False
        for state in self.planning_problem.goal.state_list:
            if state.position.contains_point(ego_state.position):
                has_reached_goal = True
                break
        return has_reached_goal
