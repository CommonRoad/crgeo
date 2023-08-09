from dataclasses import dataclass
from typing import Optional

import numpy as np
from gym.spaces import Box, Space

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.control_space.controllers.pid_controller import PIDController
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class PIDControlOptions(BaseControlSpaceOptions):
    # TODO: Allow None values, fallback to bounds imposed by vehicle model
    lower_bound_acceleration: Optional[float] = None
    upper_bound_acceleration: Optional[float] = None
    lower_bound_velocity: Optional[float] = None
    upper_bound_velocity: Optional[float] = None
    upper_bound_steering: Optional[float] = None
    lower_bound_steering: float = 0.0
    k_P_orientation: float = 0.5 # increase to increase aggresiveness
    k_D_orientation: float = 0.4 # increase to counter overshooting behavior
    k_I_orientation: float = 0.0 # increase to counter stationary error
    k_P_velocity: float = 6.0
    k_D_velocity: float = 0.5
    k_I_velocity: float = 0.0
    use_lanelet_coordinate_frame: bool = True


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
        self._lower_bound_acceleration = options.lower_bound_acceleration
        self._upper_bound_acceleration = options.upper_bound_acceleration
        self._lower_bound_velocity = options.lower_bound_velocity
        self._upper_bound_velocity = options.upper_bound_velocity

        self._upper_bound_steering = options.upper_bound_steering
        self._lower_bound_steering = options.lower_bound_steering
        self._use_lanelet_coordinate_frame = options.use_lanelet_coordinate_frame
        self._last_lanelet_orientation = 0.0

        self._pid_controller_orientation = PIDController(k_P=options.k_P_orientation,
                                                         k_D=options.k_D_orientation,
                                                         k_I=options.k_I_orientation,
                                                         d_threshold=0.0)
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
        ego_orientation = ego_vehicle.state.orientation

        v_min = self._lower_bound_velocity if self._lower_bound_velocity is not None else ego_vehicle.parameters.longitudinal.v_min
        v_max = self._upper_bound_velocity if self._upper_bound_velocity is not None else ego_vehicle.parameters.longitudinal.v_max
        a_min = self._lower_bound_acceleration if self._lower_bound_acceleration is not None else -ego_vehicle.parameters.longitudinal.a_max
        a_max = self._upper_bound_acceleration if self._upper_bound_acceleration is not None else ego_vehicle.parameters.longitudinal.a_max
        steering_min = -self._upper_bound_steering if self._upper_bound_steering is not None else ego_vehicle.parameters.steering.v_min
        steering_max = self._upper_bound_steering if self._upper_bound_steering is not None else ego_vehicle.parameters.steering.v_max

        if self._use_lanelet_coordinate_frame:
            if ego_vehicle_simulation.current_lanelet_center_polyline is not None:
                lanelet_orientation = ego_vehicle_simulation.current_lanelet_center_polyline.get_projected_direction(position)
                self._last_lanelet_orientation = lanelet_orientation
            else:
                lanelet_orientation = self._last_lanelet_orientation
            orientation = relative_orientation(lanelet_orientation, ego_orientation)
        else:
            orientation = ego_orientation

        if action[1] < 0:
            desired_velocity = -v_min * action[1]
        else:
            desired_velocity = v_max * action[1]
        desired_orientation = action[0] * np.pi / 2

        error_velocity = desired_velocity - velocity
        error_orientation = relative_orientation(orientation, desired_orientation)

        acceleration = self._pid_controller_velocity(error_velocity, dt=ego_vehicle_simulation.dt)
        steering_angle = self._pid_controller_orientation(error_orientation, dt=ego_vehicle_simulation.dt)

        acceleration = np.clip(acceleration, a_min, a_max)
        steering_angle = np.clip(steering_angle, steering_min, steering_max)
        if abs(steering_angle) < self._lower_bound_steering:
            steering_angle = 0.0

        # debug = [velocity, desired_velocity, error_velocity, acceleration, orientation, desired_orientation, error_orientation, steering_angle]
        # debug = [orientation, desired_orientation, error_orientation, steering_angle]
        # pid = self._pid_controller_orientation
        # debug = [pid._p_term, (pid._k_I * pid._i_term), (pid._k_D * pid._d_term)]
        # from commonroad_geometric.common.logging import stdout
        # stdout(", ".join([f"{x:+.2f}" for x in debug]))

        control_action = np.array([
            steering_angle,
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
