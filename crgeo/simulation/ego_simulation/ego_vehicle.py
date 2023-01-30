""" Module for managing the vehicle in the CommonRoad Gym environment
"""
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import List, Optional, Dict, Any, TYPE_CHECKING

import commonroad_dc.pycrcc as pycrcc

import numpy as np

from commonroad.common.solution import VehicleType
from commonroad.geometry.shape import Rectangle, Shape
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, FrictionCircleException
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin

from crgeo.common.geometry.helpers import make_valid_orientation
from crgeo.common.online_stats import AggregatedRunningStats
from crgeo.simulation.ego_simulation.planning_problem import EgoRoute

N_INTEGRATION_STEPS = 100


@unique
class VehicleModel(Enum):
    PM = 0
    ST = 1
    KS = 2
    MB = 3
    YawRate = 4


# Using VehicleParameterMapping from feasibility checker causes bugs
def to_vehicle_parameter(vehicle_type: VehicleType):
    if vehicle_type == VehicleType.FORD_ESCORT:
        return parameters_vehicle1()
    elif vehicle_type == VehicleType.BMW_320i:
        return parameters_vehicle2()
    elif vehicle_type == VehicleType.VW_VANAGON:
        return parameters_vehicle3()
    else:
        raise TypeError(f"Vehicle type {vehicle_type} not supported!")


def assert_vehicle_model(vehicle_model: VehicleModel):
    if vehicle_model == VehicleModel.MB:
        raise NotImplementedError(f"Vehicle model {vehicle_model} is not implemented yet!")
    else:
        return vehicle_model


class ActionBase(Enum):
    ACCELERATION = 'acceleration',
    JERK = 'jerk'

class _BaseVehicle(ABC, AutoReprMixin):
    """
    Description:
        Abstract base class of all vehicles with CommonRoad State stored in a DynamicObstacle
    """

    STATE_STATISTICS_ATTRIBUTES = ['velocity', 'yaw_rate', 'slip_angle', 'acceleration', 'steering_angle']

    def __init__(
        self,
        vehicle_type: VehicleType,
        vehicle_model: VehicleModel,
        dt: float
    ) -> None:
        """ Initialize empty object """
        self.vehicle_type = vehicle_type
        self.vehicle_model = assert_vehicle_model(vehicle_model)
        self.parameters = to_vehicle_parameter(vehicle_type)
        self.dt = dt
        self._collision_object = None
        self._vertices = np.array([
            [-self.parameters.l / 2, -self.parameters.w / 2],
            [-self.parameters.l / 2, self.parameters.w / 2],
            [self.parameters.l / 2, self.parameters.w / 2],
            [self.parameters.l / 2, -self.parameters.w / 2],
            [-self.parameters.l / 2, -self.parameters.w / 2]
        ])
        self._shape = Rectangle(
            length=self.parameters.l,
            width=self.parameters.w,
            center=np.array([0, 0])
        )
        # Contains State information
        self._dynamic_obstacle: Optional[DynamicObstacle] = None
        self._aggregate_state_statistics: Dict[str, AggregatedRunningStats] = {attr: AggregatedRunningStats() for attr in EgoVehicle.STATE_STATISTICS_ATTRIBUTES}

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def aggregate_state_statistics(self) -> Dict[str, Dict[str, float]]:
        return {attr: aggregator.asdict() for attr, aggregator in self._aggregate_state_statistics.items()}

    @property
    def shape(self) -> Shape:
        return self._shape

    # COLLISION LOGIC

    @property
    def collision_object(self) -> pycrcc.RectOBB:
        """
        Get the collision object of the vehicle

        :return: The collision object of the vehicle
        """
        return self._collision_object

    @collision_object.setter
    def collision_object(self, collision_object: pycrcc.RectOBB):
        """ Set the collision_object of the vehicle is not supported """
        raise ValueError("To set the collision_object of the vehicle directly is prohibited!")

    def create_obb_collision_object(self, state: State):
        return pycrcc.RectOBB(self.parameters.l / 2,
                              self.parameters.w / 2,
                              state.orientation,
                              state.position[0],
                              state.position[1])

    def update_collision_object(self):
        """ Updates the collision_object of the vehicle """
        self._collision_object = self.create_obb_collision_object(self.state)

    # STATE LOGIC

    @property
    def as_dynamic_obstacle(self) -> Optional[DynamicObstacle]:
        return self._dynamic_obstacle

    @property
    def obstacle_id(self) -> Optional[int]:
        return self.as_dynamic_obstacle.obstacle_id if self.as_dynamic_obstacle is not None else None

    @property
    def state(self) -> Optional[State]:
        """
        Get the current state of the vehicle

        :return: The current state of the vehicle
        """
        return self.as_dynamic_obstacle.prediction.trajectory.state_list[-1] if self.as_dynamic_obstacle is not None else None 

    @state.setter
    def state(self, state: State):
        """ Set the current state of the vehicle is not supported """
        raise ValueError("To set the state of the vehicle directly is prohibited!")

    @property
    def state_list(self) -> Optional[List[State]]:
        """
        Get the previous states of the vehicle

        :return: The previous states of the vehicle
        """
        return self.as_dynamic_obstacle.prediction.trajectory.state_list if self.as_dynamic_obstacle is not None else None

    @abstractmethod
    def update_current_state(self, **kwargs):
        """
        Update state list
        """
        raise NotImplementedError

    @property
    def current_time_step(self) -> Optional[int]:
        return self.state.time_step if self.state is not None else None

    @current_time_step.setter
    def current_time_step(self, current_time_step):
        raise ValueError("To set the current time step of the vehicle directly is prohibited!")

    def reset(self, initial_state: State) -> State:
        """
        Reset vehicle parameters.

        :param initial_state: The initial state the vehicle will be set to.
        :return: State compliant with used VehicleModel
        """
        if self.vehicle_model == VehicleModel.PM:
            orientation = initial_state.orientation if hasattr(initial_state, "orientation") else 0.0
            model_initial_state = State(**{"position": initial_state.position,
                                           "orientation": orientation,
                                           "time_step": initial_state.time_step,
                                           "velocity": initial_state.velocity * np.cos(orientation),
                                           "velocity_y": initial_state.velocity * np.sin(orientation),
                                           "acceleration": initial_state.acceleration * np.cos(orientation) if hasattr(initial_state, "acceleration") else 0.0,
                                           "acceleration_y": initial_state.acceleration * np.sin(orientation) if hasattr(initial_state, "acceleration") else 0.0})
        else:
            model_initial_state = State(**{"position": initial_state.position,
                                           "steering_angle": initial_state.steering_angle if hasattr(initial_state, "steering_angle") else 0.0,
                                           "orientation": initial_state.orientation if hasattr(initial_state, "orientation") else 0.0,
                                           "yaw_rate": initial_state.yaw_rate if hasattr(initial_state, "yaw_rate") else 0.0,
                                           "time_step": initial_state.time_step,
                                           "velocity": initial_state.velocity,
                                           "acceleration": initial_state.acceleration if hasattr(initial_state, "acceleration") else 0.0,
                                           "slip_angle": initial_state.slip_angle if hasattr(initial_state, "slip_angle") else 0.0})

        self._dynamic_obstacle = DynamicObstacle(
            obstacle_id=-1,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=self.shape,
            initial_state=model_initial_state,
            prediction=TrajectoryPrediction(
                trajectory=Trajectory(
                    initial_time_step=model_initial_state.time_step,
                    state_list=[model_initial_state]
                ),
                shape=self.shape
            )
        )
        self._dynamic_obstacle.prediction.center_lanelet_assignment = {}
        self.update_collision_object()
        for key, aggregator in self._aggregate_state_statistics.items():
            aggregator.reset()
            if hasattr(initial_state, key):
                aggregator.update(getattr(initial_state, key))
        return model_initial_state


class EgoVehicle(_BaseVehicle):
    """
    Description:
        Class for vehicle when in continuous action space respecting continuous vehicle dynamics
    """

    def __init__(
        self,
        vehicle_type: VehicleType,
        vehicle_model: VehicleModel,
        dt: float,
        ego_route: Optional[EgoRoute] = None,
    ) -> None:
        """ Initialize empty object """
        super().__init__(vehicle_type=vehicle_type, vehicle_model=vehicle_model, dt=dt)
        self._ego_route: Optional[EgoRoute] = ego_route
        self.violate_friction = False
        self.jerk_bounds = np.array([-10, 10])

        try:
            self.vehicle_dynamic = VehicleDynamics.from_model(self.vehicle_model, self.vehicle_type)
        except Exception as e:
            raise ValueError(f"Unknown vehicle model: {self.vehicle_model}") from e

    @property
    def ego_route(self) -> Optional[EgoRoute]:
        return self._ego_route

    @ego_route.setter
    def ego_route(self, value: Optional[EgoRoute]):
        self._ego_route = value

    def set_next_state(
        self,
        next_state: State
    ) -> State:
        assert self._dynamic_obstacle is not None
        self._dynamic_obstacle.prediction.trajectory.state_list.append(next_state)
        self._dynamic_obstacle.prediction.final_time_step = next_state.time_step
        self.update_collision_object()
        for key, aggregator in self._aggregate_state_statistics.items():
            if hasattr(next_state, key):
                aggregator.update(getattr(next_state, key))
        return next_state

    def update_current_state(
        self,
        action: np.ndarray,
        action_base: ActionBase,
    ) -> State:
        """Generate the next state from current state for the given action.

        :params action: rescaled action
        :params action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: next state of vehicle"""
        converted_action = self._convert_action(action, action_base)
        next_state = self._propagate_one_time_step(self.state, converted_action)
        self.set_next_state(next_state)
        return next_state

    def _convert_action(
        self,
        action: np.ndarray,
        action_base: ActionBase
    ) -> np.ndarray:
        """
        Converts action from action base to common (steering-angle and acceleration) action.

        Args:
            action (np.ndarray): different kinds of action
            action_base (np.ndarray): kind of action

        Returns:
            action, consisting of steering-angle (action[0]) and acceleration (action[1])
        """
        if action_base == ActionBase.ACCELERATION:
            assert len(action) == 2
            return action
        if action_base == ActionBase.JERK:
            steering_angle_acceleration_action = self._jerk_to_acc(action)
            assert len(steering_angle_acceleration_action) == 2
            return steering_angle_acceleration_action
        raise ValueError(f"Unknown action base: {action_base}")

    def _jerk_to_acc(self, action: np.ndarray) -> np.ndarray:
        """
        computes the acceleration based input on jerk based actions
        :param action: action based on jerk
        :return: input based on acceleration
        """
        u_input: List[float] = []

        assert self.state is not None

        if self.vehicle_model == VehicleModel.PM:
            # action[jerk_x, jerk_y]
            action = np.array([np.clip(action[0], self.jerk_bounds[0], self.jerk_bounds[1]),
                               np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input.append(self.state.acceleration + action[0] * self.dt)
            u_input.append(self.state.acceleration_y + action[1] * self.dt)

        elif self.vehicle_model == VehicleModel.KS:
            # action[steering angel speed, jerk]
            action = np.array([action[0], np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input.append(action[0])
            u_input.append(self.state.acceleration + action[1] * self.dt)

        elif self.vehicle_model == VehicleModel.YawRate:
            # action[jerk, yaw]
            action = np.array([np.clip(action[0], self.jerk_bounds[0], self.jerk_bounds[1]), action[1]])
            u_input.append(self.state.acceleration + action[0] * self.dt)
            u_input.append(action[1])
        else:
            raise ValueError(f"Unknown vehicle model: {self.vehicle_model}")

        u_input_arr = np.array(u_input)
        return u_input_arr

    def _propagate_one_time_step(
        self,
        current_state: State,
        u_input: np.ndarray,
    ) -> State:
        """Generate the next state from a given state for the given action.

        :param current_state: current state of vehicle to propagate from
        :param action: control inputs of vehicle (real input)
        :param action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: propagated state
        """

        if self.vehicle_model == VehicleModel.PM:
            # using vehicle_dynamics.state_to_array(current_state) causes error since state has orientation and velocity
            x_current = np.array([current_state.position[0],
                                  current_state.position[1],
                                  current_state.velocity,
                                  current_state.velocity_y, ])

            # if maximum absolute acceleration is exceeded, rescale the acceleration
            absolute_acc = u_input[0] ** 2 + u_input[1] ** 2
            if absolute_acc > self.parameters.longitudinal.a_max ** 2:
                rescale_factor = (self.parameters.longitudinal.a_max - 1e-6) / np.sqrt(absolute_acc)
                # rescale the acceleration to satisfy friction circle constraint
                u_input[0] *= rescale_factor
                u_input[1] *= rescale_factor

            x_next = self._forward_simulation(x_current, u_input)

            # simulated_state.acceleration = u_input[0]
            # simulated_state.acceleration_y = u_input[1]
            # simulated_state.orientation = np.arctan2(simulated_state.velocity_y, simulated_state.velocity)
            return State(
                position=np.array([x_next[0], x_next[1]]),
                velocity=x_next[2],
                velocity_y=x_next[3],
                acceleration=u_input[0],
                acceleration_y=u_input[1],
                orientation=make_valid_orientation(np.arctan2(x_next[3], x_next[2])),
                time_step=current_state.time_step + 1,
            )

        if self.vehicle_model == VehicleModel.ST:
            x_current, _ = self.vehicle_dynamic._state_to_array(current_state)
            x_next = self._forward_simulation(x_current, u_input)
            # simulated_state.acceleration = u_input[1]
            return State(
                position=np.array([x_next[0], x_next[1]]),
                steering_angle=x_next[2],
                velocity=x_next[3],
                acceleration=u_input[1],  # Changed from u_input[0] (steering_angle), see comment above
                orientation=make_valid_orientation(x_next[4]),
                yaw_rate=x_next[5],
                slip_angle=x_next[6],
                time_step=current_state.time_step + 1,
            )

        x_current = np.array([
            current_state.position[0],
            current_state.position[1],
            current_state.steering_angle,
            current_state.velocity,
            current_state.orientation,
        ])
        x_next = self._forward_simulation(x_current, u_input)

        if self.vehicle_model == VehicleModel.KS:
            # simulated_state.acceleration = u_input[1]
            # simulated_state.yaw_rate = (simulated_state.orientation - x_current_old[4]) / self.dt
            return State(
                position=np.array([x_next[0], x_next[1]]),
                steering_angle=x_next[2],
                velocity=x_next[3],
                orientation=make_valid_orientation(x_next[4]),
                acceleration=u_input[1],
                yaw_rate=(x_next[4] - x_current[4]) / self.dt,
                time_step=current_state.time_step + 1,
            )

        if self.vehicle_model == VehicleModel.YawRate:
            # simulated_state.acceleration = u_input[0]
            # simulated_state.yaw_rate = u_input[1]
            return State(
                position=np.array([x_next[0], x_next[1]]),
                steering_angle=x_next[2],
                velocity=x_next[3],
                orientation=make_valid_orientation(x_next[4]),
                acceleration=u_input[0],
                yaw_rate=u_input[1],
                time_step=current_state.time_step + 1,
            )

        raise NotImplementedError(f"VehicleModel {self.vehicle_model} not implemented")

    def _forward_simulation(
        self,
        x_current: np.ndarray,
        u_input: np.ndarray
    ) -> np.ndarray:
        """
        Forwards simulation one step, considering friction constraints.

        Args:
            x_current (np.ndarray): current state array
            u_input (np.ndarray): input action

        Returns:
            next state array
        """
        # Copy and mutate next state
        x_next = x_current.copy()
        try:
            x_next = self.vehicle_dynamic.forward_simulation(x_next, u_input, self.dt, throw=True)
            self.violate_friction = False
        except FrictionCircleException:
            self.violate_friction = True
            for _ in range(N_INTEGRATION_STEPS):
                # simulate state transition - t parameter is set to vehicle.dt but irrelevant for the current vehicle models
                # TODO：x_dot of KS model considers the action constraints, which YR and PM model have not included yet
                x_dot = np.array(self.vehicle_dynamic.dynamics(self.dt, x_next, u_input))
                # update state
                x_next = x_next + x_dot * (self.dt / N_INTEGRATION_STEPS)
        return x_next

    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        raise NotImplementedError

    def __str__(self) -> str:
        s = "EgoVehicle\n"
        s += f"- vehicle_type={self.vehicle_type}\n"
        s += f"- vehicle_model={self.vehicle_model}\n"
        s += f"- time_step={self.current_time_step}\n"
        s += f"- state={self.state}"  # state prints its own \n
        s += f"- as_dynamic_obstacle={self.as_dynamic_obstacle}"
        return s
