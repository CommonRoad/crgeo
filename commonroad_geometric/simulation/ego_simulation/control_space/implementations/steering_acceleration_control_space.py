from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from gymnasium.spaces import Box, Space

from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class SteeringAccelerationControlOptions(BaseControlSpaceOptions):
    # TODO: Allow None values, fallback to bounds imposed by vehicle model
    lower_bound_acceleration: float = -6.0
    upper_bound_acceleration: float = 6.5
    lower_bound_velocity: float = 1e-3
    lower_bound_steering: float = -0.15
    upper_bound_steering: float = 0.15
    min_velocity_steering: float = 1.0


class SteeringAccelerationSpace(BaseControlSpace):
    """
    Low-level control space for longitudinal and lateral motion planning.
    """

    def __init__(
        self,
        options: Optional[SteeringAccelerationControlOptions] = None
    ) -> None:
        options = options or SteeringAccelerationControlOptions()
        if isinstance(options, dict):
            options = SteeringAccelerationControlOptions(**options)

        self._lower_bound_acceleration = options.lower_bound_acceleration
        self._upper_bound_acceleration = options.upper_bound_acceleration
        self._lower_bound_velocity = options.lower_bound_velocity
        self._lower_bound_steering = options.lower_bound_steering
        self._upper_bound_steering = options.upper_bound_steering
        self._range_steering = (options.upper_bound_steering - options.lower_bound_steering) / 2
        self._start_steering = options.lower_bound_steering + self._range_steering
        self._min_velocity_steering = options.min_velocity_steering

        super().__init__(options)

    @property
    def gym_action_space(self) -> Space:
        return Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype="float64"
        )

    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:
        lateral_action = action[0]
        longitudinal_action = action[1]

        # TODO tanh scaling
        velocity = ego_vehicle_simulation.ego_vehicle.state.velocity
        if abs(velocity) >= self._min_velocity_steering:
            steering_angle = self._start_steering + self._range_steering * lateral_action
        else:
            steering_angle = 0.0

        if action[1] < 0:
            acceleration = -self._lower_bound_acceleration * longitudinal_action
        else:
            acceleration = self._upper_bound_acceleration * longitudinal_action

        if velocity <= self._lower_bound_velocity and acceleration < 0:
            acceleration = 0.0

        action = np.array([
            steering_angle,
            acceleration
        ])
        ego_vehicle_simulation.ego_vehicle.update_current_state(
            action=action,
            action_base=ActionBase.ACCELERATION
        )

        return True
