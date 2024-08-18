from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario


def hash_dynamic_obstacle(obstacle: DynamicObstacle) -> int:
    return hash(
        (
            obstacle.obstacle_id,
            obstacle.obstacle_role,
            obstacle.obstacle_type,
            obstacle.obstacle_shape,
            # obstacle.prediction,
            obstacle.initial_state,
            # tuple(obstacle.initial_center_lanelet_ids),
            # tuple(obstacle.initial_shape_lanelet_ids),
            # obstacle.initial_signal_state,
            # tuple(obstacle.signal_series),
        )
    )


def hash_planning_problem_set(planning_problem_set: PlanningProblemSet):
    # Used in place of broken PlanningProblemSet hash function
    return hash(
        tuple(
            (pp_id, hash_planning_problem(pp))
            for pp_id, pp in
            planning_problem_set.planning_problem_dict.items()
        )
    )


def hash_planning_problem(planning_problem: PlanningProblem):
    # Used in place of broken PlanningProblem hash function
    return hash(
        (
            planning_problem.planning_problem_id,
            planning_problem.initial_state,
            hash_goal(planning_problem.goal)
        )
    )


def hash_goal(goal: GoalRegion):
    # Used in place of broken GoalRegion hash function
    return hash(
        (
            tuple(goal.state_list),
            tuple(goal.lanelets_of_goal_position if goal.lanelets_of_goal_position else [])
        )
    )


def hash_scenario(scenario: Scenario) -> int:
    # Used in place of broken Scenario hash function
    return hash(
        (
            scenario.dt,
            tuple(scenario.scenario_id.__dict__.items()),
            scenario.lanelet_network,
            tuple(scenario.static_obstacles),
            tuple(hash_dynamic_obstacle(o) for o in scenario.dynamic_obstacles),
            # tuple(scenario.environment_obstacle),
            # tuple(scenario.phantom_obstacle),
            # scenario.author,
            # # tuple(scenario.tags),
            # scenario.affiliation,
            # scenario.source,
            # scenario.location
        )
    )
