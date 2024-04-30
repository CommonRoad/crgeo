from typing import List, Optional

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import State, CustomState, InitialState
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.lanelet import LineMarking, StopLine
from commonroad.scenario.traffic_sign import TrafficSignElement, TrafficSign, TrafficSignIDGermany
from commonroad.scenario.scenario import Scenario, Environment, TimeOfDay, Time, Underground, Weather, Location, ScenarioID

import numpy as np


def create_dummy_state(
    time_step: int = 0,
    position: Optional[np.ndarray] = None
) -> State:
    if position is None:
        position = np.array([0., 0.])
    return InitialState(
        velocity=1.0,
        acceleration=.0,
        orientation=.0,
        position=position,
        time_step=time_step
    )


def create_dummy_obstacle(
    obstacle_id: int,
    time_step: int = 0,
    lanelet: Optional[Lanelet] = None,
    dist_along_lanelet: Optional[float] = None,
) -> DynamicObstacle:
    shape = Rectangle(
        length=5.0,
        width=2.0,
        center=np.array([0, 0]),
        orientation=0.0
    )
    if lanelet is None:
        position = np.array([0., 0.])
    else:
        if dist_along_lanelet is None:
            dist_along_lanelet = 1.5
        assert 0.0 <= dist_along_lanelet <= 1.0
        position = lanelet.interpolate_position(lanelet.distance[-1] * dist_along_lanelet)[0]
    state = create_dummy_state(time_step, position)
    obstacle_kwargs = dict(
        obstacle_id=obstacle_id,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=state,
        prediction=TrajectoryPrediction(
            trajectory=Trajectory(
                initial_time_step=time_step,
                state_list=[state]
            ),
            shape=shape
        )
    )
    if lanelet is not None:
        obstacle_kwargs['initial_shape_lanelet_ids'] = {lanelet.lanelet_id}

    dummy_obstacle = DynamicObstacle(**obstacle_kwargs)

    return dummy_obstacle


def create_dummy_lanelet(lanelet_id: int, **kwargs) -> Lanelet:
    options = dict(
        right_vertices=np.array([[0, 0], [1, 0], [2, 0], [3, .5], [4, 1], [5, 1], [6, 1], [7, 0], [8, 0]]),
        left_vertices=np.array([[0, 1], [1, 1], [2, 1], [3, 1.5], [4, 2], [5, 2], [6, 2], [7, 1], [8, 1]]),
        center_vertices=np.array([[0, .5], [1, .5], [2, .5], [3, 1], [4, 1.5], [5, 1.5], [6, 1.5], [7, .5], [8, .5]]),
        lanelet_id=lanelet_id,
        predecessor=[],
        successor=[],
        adjacent_left=None,
        adjacent_right=None,
        adjacent_right_same_direction=False,
        adjacent_left_same_direction=False,
        line_marking_right_vertices=LineMarking.SOLID,
        line_marking_left_vertices=LineMarking.DASHED,
        stop_line=StopLine(start=np.array([0, 0]), end=np.array([0, 1]), line_marking=LineMarking.SOLID)
    )
    options.update(kwargs)
    return Lanelet(**options)


def create_dummy_scenario(
    lanelet_network: Optional[LaneletNetwork] = None,
    dynamic_obstacles: Optional[List[DynamicObstacle]] = None
) -> Scenario:
    dynamic_obstacles = dynamic_obstacles or []
    environment = Environment(Time(12, 15), TimeOfDay.NIGHT, Weather.SNOW, Underground.ICE)
    location = Location(geo_name_id=123, gps_latitude=456, gps_longitude=789, environment=environment)
    scenario = Scenario(0.1, location=location, scenario_id=ScenarioID())
    if lanelet_network is None:
        lanelet_network = LaneletNetwork.create_from_lanelet_list([create_dummy_lanelet(0), create_dummy_lanelet(1)])
    scenario.add_objects(lanelet_network)
    for obstacle in dynamic_obstacles:
        scenario.add_objects(obstacle)
    return scenario
