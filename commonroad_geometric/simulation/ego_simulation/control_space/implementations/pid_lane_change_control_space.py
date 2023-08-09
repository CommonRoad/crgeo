from dataclasses import dataclass
from typing import Optional

import numpy as np
from gym.spaces import Box, Space

from commonroad_geometric.common.geometry.helpers import make_valid_orientation, relative_orientation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.control_space.controllers.pid_controller import PIDController
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class PIDLaneChangeControlOptions(BaseControlSpaceOptions):
    # TODO: Allow None values, fallback to bounds imposed by vehicle model
    lower_bound_acceleration: Optional[float] = None
    upper_bound_acceleration: Optional[float] = None
    lower_bound_velocity: Optional[float] = None
    upper_bound_velocity: Optional[float] = None
    upper_bound_steering: Optional[float] = None
    lower_bound_steering: float = 0.0
    k_P_orientation: float = 0.5 # increase to increase aggresiveness
    k_D_orientation: float = 0.4 # increase to counter overshooting behavior
    k_I_orientation: float = 0.1 # increase to counter stationary error
    k_P_velocity: float = 6.0
    k_D_velocity: float = 0.5
    k_I_velocity: float = 0.0
    finish_action_threshold: float = 0.02
    use_lanelet_coordinate_frame: bool = True


class PIDLaneChangeControlSpace(BaseControlSpace):
    """
    High level lane change control space implemented with PID-based 
    low-level motion planning.
    """
    def __init__(
        self,
        options: Optional[PIDLaneChangeControlOptions] = None
    ) -> None:
        options = options or PIDLaneChangeControlOptions()
        self._lower_bound_acceleration = options.lower_bound_acceleration
        self._upper_bound_acceleration = options.upper_bound_acceleration
        self._lower_bound_velocity = options.lower_bound_velocity
        self._upper_bound_velocity = options.upper_bound_velocity

        self._upper_bound_steering = options.upper_bound_steering
        self._lower_bound_steering = options.lower_bound_steering
        self._use_lanelet_coordinate_frame = options.use_lanelet_coordinate_frame
        self._last_lanelet_orientation = 0.0
        self._desired_lanelet_polyline: Optional[ContinuousPolyline] = None
        self._finish_action_threshold = options.finish_action_threshold

        self._pid_controller_orientation = PIDController(k_P=options.k_P_orientation,
                                                         k_D=options.k_D_orientation,
                                                         k_I=options.k_I_orientation,
                                                         d_threshold=0.0)
        self._pid_controller_velocity = PIDController(k_P=options.k_P_velocity,
                                                      k_D=options.k_D_velocity,
                                                      k_I=options.k_I_velocity)
        self._lateral_error_prior = None
        self._subsubstep = 0
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

        if ego_vehicle_simulation.current_lanelet_center_polyline is not None:
            lanelet_orientation = ego_vehicle_simulation.current_lanelet_center_polyline.get_projected_direction(position)
            self._last_lanelet_orientation = lanelet_orientation
        else:
            lanelet_orientation = self._last_lanelet_orientation
        orientation = relative_orientation(lanelet_orientation, ego_orientation)

        if action[1] < 0:
            desired_velocity = v_min if v_min * action[1] < v_min else v_min * action[1]
        else:
            desired_velocity = v_max if v_max * action[1] > v_max else v_min if v_max * action[1] < v_min else v_max * action[1]

        if self._subsubstep == 0 and substep_index == 0:
            self._lateral_error_prior = None
            desired_lanechange = action[0]
            desired_lanechange_disc = -1 if desired_lanechange < -0.5 else (1 if desired_lanechange > 0.5 else 0)
            current_lanelet = ego_vehicle_simulation.current_lanelets[0]

            if desired_lanechange_disc == 1 and current_lanelet.adj_left_same_direction:
                desired_lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(current_lanelet.adj_left)

            elif desired_lanechange_disc == -1 and current_lanelet.adj_right_same_direction:
                desired_lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(current_lanelet.adj_right)

            else:
                desired_lanelet = None

            # print(desired_lanechange_disc, current_lanelet.lanelet_id)

            if desired_lanelet is not None:
                self._desired_lanelet = desired_lanelet
                self._desired_lanelet_polyline = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(desired_lanelet.lanelet_id)
            else:
                self._desired_lanelet_polyline = None
                self._desired_lanelet = None

        if self._desired_lanelet_polyline is not None:
            lateral_error = self._desired_lanelet_polyline.get_lateral_distance(position)
        else:
            lateral_error = 0.0
        current_lanelet_id = ego_vehicle_simulation.current_lanelets[0].lanelet_id
        desired_orientation = make_valid_orientation(-lateral_error/20)
        error_velocity = desired_velocity - velocity
        error_orientation = relative_orientation(orientation, desired_orientation)

        acceleration = self._pid_controller_velocity(error_velocity, dt=ego_vehicle_simulation.dt)
        steering_angle = self._pid_controller_orientation(error_orientation, dt=ego_vehicle_simulation.dt)

        acceleration = np.clip(acceleration, a_min, a_max)
        if self._desired_lanelet_polyline is not None:
            print((f'orientation: {orientation:+.4f}, d_orientation: {desired_orientation:+.4f}, e_o {error_orientation:+.4f}, l_e: {lateral_error:+.4f}, s_a_c: {steering_angle:+.4f}, l_id: {ego_vehicle_simulation.current_lanelets[0].lanelet_id}, d_l_id: {self._desired_lanelet.lanelet_id}, disc_r: {self._desired_lanelet.adj_right}, disc_l: {self._desired_lanelet.adj_left}'))
        steering_angle = np.clip(steering_angle, steering_min, steering_max)

        if self._desired_lanelet is not None and self._desired_lanelet.adj_left == current_lanelet_id:
            steering_angle = steering_angle

        if abs(steering_angle) < self._lower_bound_steering or abs(lateral_error) <= self._finish_action_threshold or self._desired_lanelet_polyline is None:
            steering_angle = 0.0

        # debug = [velocity, desired_velocity, error_velocity, acceleration, orientation, desired_orientation, error_orientation, steering_angle]
        

        control_action = np.array([
            steering_angle,
            acceleration
        ], dtype=np.float64)
        ego_vehicle.update_current_state(
            action=control_action,
            action_base=ActionBase.ACCELERATION
        )


        finished_action = self._desired_lanelet_polyline is None or ((abs(lateral_error) < self._finish_action_threshold or self._desired_lanelet.lanelet_id == current_lanelet_id) and (self._desired_lanelet_polyline is not None) or self._desired_lanelet.lanelet_id == current_lanelet_id)
        # if (abs(lateral_error) < self._finish_action_threshold) and (self._desired_lanelet_polyline is not None):
        #     print(f'substep_idx: {substep_index}, f_a: {finished_action}')
        debug = [orientation, desired_orientation, error_orientation, steering_angle, finished_action, lateral_error, substep_index]

        if self._lateral_error_prior is not None and abs(self._lateral_error_prior) < abs(lateral_error) and self._desired_lanelet.lanelet_id == current_lanelet_id:
            finished_action = True
        self._lateral_error_prior = lateral_error
        from commonroad_geometric.common.logging import stdout
        stdout(", ".join([f"{x:+.2f}" for x in debug]))
        self._subsubstep += 1
        if finished_action:
            self._subsubstep = 0
        print(f'substep_idx: {substep_index}, f_a: {finished_action}')        

        return True

    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        self._last_lanelet_orientation = 0.0
        self._pid_controller_orientation.clear()
        self._pid_controller_velocity.clear()
        self._desired_lanelet_polyline = None
        self._desired_lanelet = None
        self._lateral_error_prior = None
        self._subsubstep = 0
