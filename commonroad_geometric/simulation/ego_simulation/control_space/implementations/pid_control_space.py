from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Space

from commonroad_geometric.common.geometry.helpers import relative_orientation, make_valid_orientation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.utils.pid_controller import PIDController
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def shortest_angle_diff(target_angle, current_angle):
    """Calculate the shortest difference between two angles, considering wrap-around."""
    normalized_diff = normalize_angle(target_angle - current_angle)
    return normalized_diff

def calculate_circular_difference(angle1, angle2):
    """Calculate the shortest circular difference between two angles."""
    diff = normalize_angle(angle1 - angle2)
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return diff

def wrap_orientation(orientation):
    """Wrap the orientation to stay within [-pi, pi] range, handling boundary crossing explicitly."""
    if orientation > np.pi:
        return orientation - 2 * np.pi
    elif orientation < -np.pi:
        return orientation + 2 * np.pi
    return orientation

@dataclass
class PIDControlOptions(BaseControlSpaceOptions):
    lower_bound_acceleration: float = -10.0
    upper_bound_acceleration: float = 6.5
    lower_bound_velocity: float = 1e-3
    upper_bound_velocity: Optional[float] = None
    lower_bound_steering: float = -0.4
    upper_bound_steering: float = 0.4

    min_velocity_steering: float = 1.0
    k_P_orientation: float = 5.0  # increase to increase aggresiveness
    k_D_orientation: float = 0.0  # increase to counter overshooting behavior
    k_I_orientation: float = 0.0  # increase to counter stationary error
    k_yaw_rate: float = 1.75

    windup_guard_orientation: float = 10.0
    k_P_velocity: float = 7.5
    k_D_velocity: float = 0.5
    k_I_velocity: float = 0.0

    use_lanelet_coordinate_frame: bool = False
    steering_coefficient: float = 1.5
    steering_error_clipping: float = np.pi/6


class PIDControlSpace(BaseControlSpace):
    """
    Low-level space where the contol actions correspond to
    setting the reference setpoints for longitudinal and lateral PID controllers.
    """

    def __init__(
        self,
        options: Optional[PIDControlOptions] = None
    ) -> None:
        options = options or PIDControlOptions()
        if isinstance(options, dict):
            options = PIDControlOptions(**options)
        self._options = options
        self._lower_bound_acceleration = options.lower_bound_acceleration
        self._upper_bound_acceleration = options.upper_bound_acceleration
        self._lower_bound_velocity = options.lower_bound_velocity
        self._upper_bound_velocity = options.upper_bound_velocity

        self._upper_bound_steering = options.upper_bound_steering
        self._lower_bound_steering = options.lower_bound_steering
        self._use_lanelet_coordinate_frame = options.use_lanelet_coordinate_frame
        self._last_lanelet_orientation = 0.0
        self._desired_orientation = 0.0

        self._pid_controller_orientation = PIDController(k_P=options.k_P_orientation,
                                                         k_D=options.k_D_orientation,
                                                         k_I=options.k_I_orientation,
                                                         windup_guard=options.windup_guard_orientation,
                                                         d_threshold=2.0)
        self._pid_controller_velocity = PIDController(k_P=options.k_P_velocity,
                                                      k_D=options.k_D_velocity,
                                                      k_I=options.k_I_velocity)

        super().__init__(options)

    @property
    def gym_action_space(self) -> Space:
        return Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype="float64"
        )

    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:
        # TODO tanh scaling
        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        position = ego_vehicle.state.position
        velocity = ego_vehicle.state.velocity
        yaw_rate = ego_vehicle.state.yaw_rate
        ego_orientation = ego_vehicle.state.orientation
        steering_angle = ego_vehicle.state.steering_angle

        v_min = self._lower_bound_velocity if self._lower_bound_velocity is not None else ego_vehicle.parameters.longitudinal.v_min
        v_max = self._upper_bound_velocity if self._upper_bound_velocity is not None else ego_vehicle.parameters.longitudinal.v_max
        a_min = self._lower_bound_acceleration if self._lower_bound_acceleration is not None else -ego_vehicle.parameters.longitudinal.a_max
        a_max = self._upper_bound_acceleration if self._upper_bound_acceleration is not None else ego_vehicle.parameters.longitudinal.a_max
        steering_min = self._lower_bound_steering if self._lower_bound_steering is not None else ego_vehicle.parameters.steering.v_min
        steering_max = self._upper_bound_steering if self._upper_bound_steering is not None else ego_vehicle.parameters.steering.v_max

        if self._use_lanelet_coordinate_frame:
            if ego_vehicle_simulation.current_lanelet_center_polyline is not None:
                lanelet_orientation = ego_vehicle_simulation.current_lanelet_center_polyline.get_projected_direction(
                    position)
                self._last_lanelet_orientation = lanelet_orientation
            else:
                lanelet_orientation = self._last_lanelet_orientation
            orientation = relative_orientation(lanelet_orientation, ego_orientation)
            self._desired_orientation = action[0] * np.pi / 2
        else:
            orientation = ego_orientation
            # self._desired_orientation += ego_vehicle_simulation.dt * action[0] * self._options.steering_coefficient
            self._desired_orientation = orientation + action[0] * np.pi / 2 * self._options.steering_coefficient
            self._desired_orientation = wrap_orientation(self._desired_orientation)

            # Calculate difference
            difference = calculate_circular_difference(self._desired_orientation, orientation)

            # Restrict difference to within +- self._options.steering_orientation_threshold
            if difference > self._options.steering_error_clipping:
                self._desired_orientation = orientation + self._options.steering_error_clipping
            elif difference < -self._options.steering_error_clipping:
                self._desired_orientation = orientation - self._options.steering_error_clipping
            self._desired_orientation = normalize_angle(self._desired_orientation)

        # Calculate the error as before, assuming the normalization is done within the error calculation
        error_orientation = calculate_circular_difference(self._desired_orientation, orientation)
        error_orientation = np.clip(error_orientation, -self._options.steering_error_clipping, self._options.steering_error_clipping)
        error_steering_angle = calculate_circular_difference(error_orientation, steering_angle)

        if action[1] < 0:
            desired_velocity = -v_min * action[1]
        else:
            desired_velocity = v_max * action[1]
        error_velocity = desired_velocity - velocity

        acceleration = self._pid_controller_velocity(error_velocity, dt=ego_vehicle_simulation.dt)
        acceleration = np.clip(acceleration, a_min, a_max)

        steering_angle_speed = self._pid_controller_orientation(error_steering_angle, dt=ego_vehicle_simulation.dt) - self._options.k_yaw_rate*yaw_rate
        steering_angle_speed = np.clip(steering_angle_speed, steering_min, steering_max)

        p_term = self._pid_controller_orientation._p_term
        d_term = self._pid_controller_orientation._d_term
        i_term = self._pid_controller_orientation._i_term
        # print(f"{steering_angle_speed=:.4f}, {p_term=:.4f}, {d_term=:.4f}, {i_term=:.4f} {yaw_rate=:.4f}")
        # print(f"{velocity=}, {desired_velocity=}, {acceleration=}")

        control_action = np.array([
            steering_angle_speed,
            acceleration
        ], dtype=np.float64)
        ego_vehicle.update_current_state(
            action=control_action,
            action_base=ActionBase.ACCELERATION
        )
        return True

    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        self._last_lanelet_orientation = 0.0
        self._pid_controller_orientation.clear()
        self._pid_controller_velocity.clear()
