import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import Occupancy, Prediction, TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State
from commonroad.scenario.trajectory import Trajectory




def interval__eq__(self: Interval, other: Interval) -> bool:
    if not isinstance(other, Interval):
        return False

    return self._start == other.start and self._end == other.end


def rectangle__eq__(self: Rectangle, other: Rectangle) -> bool:
    if not isinstance(other, Rectangle):
        return False

    center_string = np.array2string(np.around(self._center.astype(float), 10), precision=10)
    center_other_string = np.array2string(np.around(other.center.astype(float), 10), precision=10)

    return (
        self._length == other.length
        and self._width == other.width
        and center_string == center_other_string
        and self._orientation == other.orientation
    )


def state__eq__(self: State, other: State) -> bool:
    if not isinstance(other, State):
        return False
    if set(self.attributes) != set(other.attributes):
        return False

    dec = 10
    for attr in self.attributes:
        val_self = getattr(self, attr)
        val_other = getattr(other, attr)

        if attr == "position" and (isinstance(val_self, np.ndarray) or isinstance(val_other, np.ndarray)):
            if isinstance(val_self, np.ndarray) and isinstance(val_other, np.ndarray):
                val_self = tuple(np.around(self.position.astype(float), dec))
                val_other = tuple(np.around(self.position.astype(float), dec))
            else:
                return False

        if isinstance(val_self, float):
            val_self = round(val_self, dec)
        if isinstance(val_other, float):
            val_other = round(val_other, dec)

        if val_self != val_other:
            return False

    return True


def trajectory__eq__(self: Trajectory, other: Trajectory) -> bool:
    if not isinstance(other, Trajectory):
        return False

    return self._initial_time_step == other.initial_time_step and list(self._state_list) == list(other.state_list)


def trajectory_prediction__eq__(self: TrajectoryPrediction, other: TrajectoryPrediction) -> bool:
    prediction_eq = self._shape == other.shape and self._trajectory == other.trajectory and \
                    self.center_lanelet_assignment == other.center_lanelet_assignment and \
                    self.shape_lanelet_assignment == other.shape_lanelet_assignment and Prediction.__eq__(self, other)

    return prediction_eq


def occupancy__eq__(self: Occupancy, other: Occupancy) -> bool:
    if not isinstance(other, Occupancy):
        return False

    return self._time_step == other.time_step and self._shape == other.shape


def prediction__eq__(self: Prediction, other: Prediction) -> bool:
    if not isinstance(other, Prediction):
        return False

    return self._initial_time_step == other.initial_time_step and self._occupancy_set == other.occupancy_set


def obstacle__eq__(self: Obstacle, other: Obstacle) -> bool:
    if not isinstance(other, Obstacle):
        return False

    initial_center_lanelet_ids = (
        list() if self._initial_center_lanelet_ids is None else list(self._initial_center_lanelet_ids)
    )
    initial_center_lanelet_ids_other = (
        list() if other.initial_center_lanelet_ids is None else list(other.initial_center_lanelet_ids)
    )

    initial_shape_lanelet_ids = (
        list() if self._initial_shape_lanelet_ids is None else list(self._initial_shape_lanelet_ids)
    )
    initial_shape_lanelet_ids_other = (
        list() if other.initial_shape_lanelet_ids is None else list(other.initial_shape_lanelet_ids)
    )

    return (
        self._obstacle_id == other.obstacle_id
        and self._obstacle_role == other.obstacle_role
        and self._obstacle_type == other.obstacle_type
        and self._obstacle_shape == other.obstacle_shape
        and self._initial_state == other.initial_state
        and initial_center_lanelet_ids == initial_center_lanelet_ids_other
        and initial_shape_lanelet_ids == initial_shape_lanelet_ids_other
        and self._initial_signal_state == other.initial_signal_state
        and self._signal_series == other.signal_series
    )


def dynamic_obstacle__eq__(self: DynamicObstacle, other: DynamicObstacle) -> bool:
    if not isinstance(other, DynamicObstacle):
        return False

    return Prediction.__eq__(self._prediction, other.prediction) and Obstacle.__eq__(self, other)


def planning_problem_set__eq__(self: PlanningProblemSet, other: PlanningProblemSet) -> bool:
    if not isinstance(other, PlanningProblemSet):
        return False

    return self.planning_problem_dict.items() == other.planning_problem_dict.items()


def planning_problem__eq__(self: PlanningProblem, other: PlanningProblem) -> bool:
    if not isinstance(other, PlanningProblem):
        return False

    return (
        self.planning_problem_id == other.planning_problem_id
        and self.initial_state == other.initial_state
        and self.goal == other.goal
    )


def goal_region__eq__(self: GoalRegion, other: GoalRegion) -> bool:
    if not isinstance(other, GoalRegion):
        return False

    lanelets_of_goal_position = (
        None if self.lanelets_of_goal_position is None else self.lanelets_of_goal_position.items()
    )
    lanelets_of_goal_position_other = (
        None if other.lanelets_of_goal_position is None else other.lanelets_of_goal_position.items()
    )
    return self.state_list == other.state_list and lanelets_of_goal_position == lanelets_of_goal_position_other


def scenario__eq__(self: Scenario, other: Scenario) -> bool:
    if not isinstance(other, Scenario):
        return False

    return (
        str(self.dt) == str(other.dt)
        and self.scenario_id == other.scenario_id
        and self.lanelet_network == other.lanelet_network
        and self.static_obstacles == other.static_obstacles
        and self.dynamic_obstacles == other.dynamic_obstacles
        and self.environment_obstacle == other.environment_obstacle
        and self.phantom_obstacle == other.phantom_obstacle
        and self.author == other.author
        and self.tags == other.tags
        and self.affiliation == other.affiliation
        and self.source == other.source
        and self.location == other.location
    )


def monkey_patch__eq__() -> None:
    # Monkey-patching equality operator for tests
    Interval.__eq__ = interval__eq__
    Rectangle.__eq__ = rectangle__eq__
    State.__eq__ = state__eq__
    Trajectory.__eq__ = trajectory__eq__
    TrajectoryPrediction.__eq__ = trajectory_prediction__eq__
    Occupancy.__eq__ = occupancy__eq__
    Prediction.__eq__ = prediction__eq__
    Obstacle.__eq__ = obstacle__eq__
    DynamicObstacle.__eq__ = dynamic_obstacle__eq__
    PlanningProblemSet.__eq__ = planning_problem_set__eq__
    PlanningProblem.__eq__ = planning_problem__eq__
    GoalRegion.__eq__ = goal_region__eq__
    Scenario.__eq__ = scenario__eq__
