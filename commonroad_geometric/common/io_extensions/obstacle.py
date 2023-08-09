import warnings
from typing import List, Literal, Optional, overload

from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, State


def map_obstacle_type(obstacle_type: ObstacleType) -> int:
    try:
        return {
            ObstacleType.CAR: 0,
            ObstacleType.TRUCK: 1,
            ObstacleType.BUS: 2,
            ObstacleType.MOTORCYCLE: 3
        }[obstacle_type]
    except KeyError:
        warnings.warn(f"Unknown obstacle type {obstacle_type} encountered")
        return 4


def get_obstacle_lanelet_assignment(
    obstacle: DynamicObstacle,
    time_step: int,
    use_center_lanelet_assignment: bool = True,
    use_shape_lanelet_assignment: bool = True,
) -> List[int]:
    if use_center_lanelet_assignment:
        center_assignment = obstacle.prediction.center_lanelet_assignment
        if center_assignment is not None and time_step in center_assignment:
            assignment = center_assignment[time_step]
            return assignment if isinstance(assignment, list) else list(assignment)

    if use_shape_lanelet_assignment:
        shape_assignment = obstacle.prediction.shape_lanelet_assignment
        if shape_assignment is not None and time_step in shape_assignment:
            assignment = shape_assignment[time_step]
            return assignment if isinstance(assignment, list) else list(assignment)

    return []

# DynamicObstacle.__hash__ = lambda o: hash(o.obstacle_id) # type: ignore


class ObstacleUndefinedStateException(ValueError):
    def __init__(self, obstacle: DynamicObstacle, time_step: int) -> None:
        message = f"Obstacle {obstacle.obstacle_id} has undefined state at time-step {time_step}"
        self.message = message
        super(ObstacleUndefinedStateException, self).__init__(message)


@overload
def state_at_time(obstacle: DynamicObstacle, time_step: int, assume_valid: Literal[True]) -> State:
    ...

@overload
def state_at_time(obstacle: DynamicObstacle, time_step: int, assume_valid: bool = False) -> Optional[State]:
    ...


def state_at_time(obstacle: DynamicObstacle, time_step: int, assume_valid: bool = False) -> Optional[State]:
    """
    Returns the predicted state of the obstacle at a specific time step.

    :param time_step: discrete time step
    :return: predicted state of the obstacle at time step
    """
    if time_step == obstacle.initial_state.time_step:
        return obstacle.initial_state
        
    trajectory = obstacle.prediction.trajectory
    if assume_valid:
        try:
            return trajectory._state_list[time_step - trajectory._initial_time_step]
        except IndexError:
            raise ObstacleUndefinedStateException(obstacle, time_step)
    elif (
        trajectory._initial_time_step
        <= time_step
        < trajectory._initial_time_step + len(trajectory._state_list)
    ):
        return trajectory._state_list[time_step - trajectory._initial_time_step]
    return None


def get_state_list(obstacle: DynamicObstacle, upper: Optional[int] = None) -> List[State]:
    states = []
    upper_bnd = obstacle.prediction.final_time_step + 1
    if upper is not None:
        upper_bnd = min(upper_bnd, upper + 1)
    for time_step in range(obstacle.initial_state.time_step, upper_bnd):
        states.append(state_at_time(obstacle, time_step, assume_valid=True))
    return states
