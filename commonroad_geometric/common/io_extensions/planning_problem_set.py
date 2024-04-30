__desc__ = """
    Taken from commonroad-dataset-converters
"""

import random
from enum import Enum
from typing import Optional, Type, Union

import numpy as np
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State


class NoCarException(Exception):
    pass


class Routability(Enum):
    ANY = 0
    # REGULAR_ANDREVERSED = 1
    REGULAR_STRICT = 2

    @classmethod
    def options(cls):
        return [type(item) for item in cls]


class ObstacleToPlanningProblemException(ValueError):
    pass


def obstacle_to_planning_problem(
    obstacle: DynamicObstacle, lanelet_network: LaneletNetwork,
    planning_problem_id: int, final_time_step: Optional[int] = None,
    orientation_half_range: float = 0.2,
    velocity_half_range: float = 10, time_step_half_range: int = 25,
    random_start_offset: bool = False, max_timesteps: Optional[Union[int, float]] = None,
    min_timesteps: Optional[int] = None, min_distance: Optional[float] = None, max_distance: Optional[float] = None,
    no_entry_lanelets: bool = False
) -> PlanningProblem:
    """
    Generates planning problem using initial and final states of a DynamicObstacle
    """
    dynamic_obstacle_shape = obstacle.obstacle_shape
    dynamic_obstacle_state_list = obstacle.prediction.trajectory.state_list

    if max_timesteps is None:
        max_timesteps = len(dynamic_obstacle_state_list)
    elif isinstance(max_timesteps, float):
        max_timesteps = int(max_timesteps * len(dynamic_obstacle_state_list))

    if random_start_offset:
        start_index = random.randint(
            0,
            len(dynamic_obstacle_state_list) // 2
        )
        if max_timesteps is not None:
            end_index = min(start_index + max_timesteps, len(dynamic_obstacle_state_list) - 1)
        else:
            end_index = len(dynamic_obstacle_state_list) - 1
    else:
        start_index = 0
        end_index = len(dynamic_obstacle_state_list) - 1

    if min_timesteps is not None and (end_index - start_index) < min_timesteps:
        raise ObstacleToPlanningProblemException()

    start_state = dynamic_obstacle_state_list[start_index]
    end_state = dynamic_obstacle_state_list[end_index]

    if no_entry_lanelets:
        start_lanelets = lanelet_network.find_lanelet_by_position([start_state.position])[0]
        for lanelet_id in start_lanelets:
            if not lanelet_network.find_lanelet_by_id(lanelet_id).predecessor:
                raise ObstacleToPlanningProblemException()

    goal_distance = np.linalg.norm(start_state.position - end_state.position)
    if min_distance is not None and goal_distance < min_distance:
        raise ObstacleToPlanningProblemException()
    if max_distance is not None and goal_distance > max_distance:
        raise ObstacleToPlanningProblemException()

    # define orientation, velocity and time step intervals as goal region
    orientation_interval = AngleInterval(end_state.orientation - orientation_half_range,
                                         end_state.orientation + orientation_half_range)
    velocity_interval = Interval(end_state.velocity - velocity_half_range,
                                 end_state.velocity + velocity_half_range)
    if final_time_step is None:
        final_time_step = end_state.time_step + time_step_half_range

    time_step_interval = Interval(0, final_time_step)

    goal_shape = Rectangle(length=dynamic_obstacle_shape.length + 2.0,
                           width=max(dynamic_obstacle_shape.width + 1.0, 3.5),
                           center=end_state.position,
                           orientation=end_state.orientation)

    # find goal lanelet
    goal_lanelets = lanelet_network.find_lanelet_by_position([end_state.position])
    if len(goal_lanelets[0]) == 0:
        raise ObstacleToPlanningProblemException(
            "Selected final state for planning problem is out of road. Skipping this scenario")

    goal_lanelet_id = goal_lanelets[0][0]
    goal_lanelet = lanelet_network.find_lanelet_by_id(goal_lanelet_id)
    goal_lanelet_polygon = goal_lanelet.convert_to_polygon()
    if goal_lanelet_polygon.shapely_object.area > goal_shape.shapely_object.area:
        goal_position = goal_lanelet_polygon
    else:
        goal_position = goal_shape

    goal_state = State(
        position=goal_position,
        orientation=orientation_interval,
        velocity=velocity_interval,
        time_step=time_step_interval
    )
    goal_region = GoalRegion(state_list=[goal_state])

    start_state.yaw_rate = 0.0
    start_state.slip_angle = 0.0

    return PlanningProblem(planning_problem_id, start_state, goal_region)


def check_routability_planning_problem(
    scenario: Scenario, planning_problem: PlanningProblem,
    max_difficulity: Type[Routability]
) -> bool:
    """
    Checks if a planning problem is routable on scenario
    :param scenario: CommonRoad scenario
    :param planning_problem: Planning Problem to be solved
    :param max_difficulity: difficulty until which planing problem is considered routable.
        Routability.ANY: dont do any checks, always return True
        Routability.REGULAR_STRICT: only return True if default route planner can find a route

    :return: bool, True if CommonRoad planning problem is routeable with max_difficulity
    """
    from commonroad_geometric.external.commonroad_route_planner.route_planner import RoutePlanner

    if max_difficulity == Routability.ANY:
        return True

    elif max_difficulity == Routability.REGULAR_STRICT:
        route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
        candidate_holder = route_planner.plan_routes()
        _, num_candidates = candidate_holder.retrieve_all_routes()

        if num_candidates > 0:
            return True  # there are some routes.
        else:
            return False

    else:
        raise ValueError(f"option not defined: {max_difficulity}")
