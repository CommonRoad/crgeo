from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, sin
from typing import Optional, TYPE_CHECKING

import numpy as np
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.geometry.helpers import relative_orientation

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class LookAheadPathObservation:
    crosstrack_error: float
    yaw_diff: float
    yaw_rate_diff: float
    yaw_rate_path: float

@dataclass
class PathObservation:
    yaw_diff_front: float
    yaw_diff_center: float
    yaw_rate_diff_front: float
    crosstrack_error: float
    yaw_rate_front: float
    yaw_rate_path_front: float
    look_ahead_observations: list[LookAheadPathObservation]


def flatten_path_observation(observation: PathObservation) -> np.ndarray:
    flat_list = [
        observation.yaw_diff_front,
        observation.yaw_diff_center,
        observation.yaw_rate_diff_front,
        observation.crosstrack_error,
        observation.yaw_rate_front,
        observation.yaw_rate_path_front,
    ]
    
    for look_ahead_obs in observation.look_ahead_observations:
        flat_list.extend([
            look_ahead_obs.crosstrack_error,
            look_ahead_obs.yaw_diff,
            look_ahead_obs.yaw_rate_diff,
            look_ahead_obs.yaw_rate_path,
        ])
    
    return np.array(flat_list)


class PathObserver:
    """
    Provides a data struct with navigational features that relate
    the current ego vehicle pose to the desired path, now supporting
    multiple look-ahead distances.
    """

    def __init__(
        self,
        look_ahead_distances: list[float] = [15.0],
        look_ahead_dynamic_offset: float = 0.0
    ) -> None:
        self.look_ahead_distances = look_ahead_distances
        self.look_ahead_dynamic_offset = look_ahead_dynamic_offset
        self._last_arclength: Optional[float] = None
        self._last_yaw_diff_look_ahead = {d: None for d in look_ahead_distances}
        self._last_yaw_path_look_ahead = {d: None for d in look_ahead_distances}
        self._last_yaw_rate_look_ahead = {d: None for d in look_ahead_distances}
        self._last_yaw_diff_front: Optional[float] = None
        self._last_yaw_path_front: Optional[float] = None
        self._last_yaw_rate_front: Optional[float] = None
        self._last_steering_angle: Optional[float] = None

    def observe(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> PathObservation:
        """
        Updates the state variables and returns the updated path observation data class object.

        Args:
            ego_vehicle_simulation (EgoVehicleSimulation): Current simulation instance.
        """
        ego_parameters = ego_vehicle_simulation.ego_vehicle.parameters
        current_state = ego_vehicle_simulation.ego_vehicle.state
        yaw = current_state.orientation
        v = current_state.velocity

        path = ego_vehicle_simulation.ego_route.planning_problem_path_polyline
        assert path is not None

        p_center = current_state.position
        p_front = p_center + ego_parameters.a*np.array([cos(yaw), sin(yaw)])

        p_center_arclength = path.get_projected_arclength(
            p_center,
            linear_projection=False
        )
        p_front_arclength = path.get_projected_arclength(
            p_front,
            linear_projection=False
        )

        yaw_path_center = path.get_direction(p_center_arclength)
        yaw_path_front = path.get_direction(p_front_arclength)
        yaw_diff_center = relative_orientation(yaw_path_center, yaw)
        yaw_diff_front = relative_orientation(yaw_path_front, yaw)

        if self._last_yaw_diff_front is not None:
            yaw_rate_front = relative_orientation(self._last_yaw_diff_front, yaw_diff_front) / ego_vehicle_simulation.dt
        else:
            yaw_rate_front = 0.0
        self._last_yaw_diff_front = yaw_diff_front
        if self._last_yaw_path_front is not None:
            yaw_rate_path_front = relative_orientation(self._last_yaw_path_front, yaw_path_front) / ego_vehicle_simulation.dt
        else:
            yaw_rate_path_front = 0.0
        self._last_yaw_path_front = yaw_path_front
        yaw_rate_diff_front = yaw_rate_front - self._last_yaw_rate_front if self._last_yaw_rate_front is not None else 0.0
        self._last_yaw_rate_front = yaw_rate_front

        # p_proj_center = path(p_center_arclength)
        p_proj_front = path(p_front_arclength, use_cached=False)

        crosstrack_error = path.get_lateral_distance(
            p_center,
            linear_projection=False
        )

        look_ahead_observations = []
        for look_ahead_distance in self.look_ahead_distances:
            dynamic_look_ahead_distance = look_ahead_distance + self.look_ahead_dynamic_offset*v
            p_look_ahead_extrapolation = p_front + dynamic_look_ahead_distance*np.array([cos(yaw), sin(yaw)])

            p_look_ahead_arclength, p_look_ahead = ego_vehicle_simulation.ego_route.set_look_ahead_point(
                look_ahead_distance,
                p_look_ahead_extrapolation,
                linear_projection=False
            )

            yaw_path_look_ahead = path.get_direction(p_look_ahead_arclength)
            yaw_diff_look_ahead = relative_orientation(yaw_path_look_ahead, yaw)

            if self._last_yaw_diff_look_ahead[look_ahead_distance] is not None:
                yaw_rate_look_ahead = relative_orientation(self._last_yaw_diff_look_ahead[look_ahead_distance], yaw_diff_look_ahead) / ego_vehicle_simulation.dt
            else:
                yaw_rate_look_ahead = 0.0

            self._last_yaw_diff_look_ahead[look_ahead_distance] = yaw_diff_look_ahead

            if self._last_yaw_path_look_ahead[look_ahead_distance] is not None:
                yaw_rate_path_look_ahead = relative_orientation(self._last_yaw_path_look_ahead[look_ahead_distance], yaw_path_look_ahead) / ego_vehicle_simulation.dt
            else:
                yaw_rate_path_look_ahead = 0.0
            self._last_yaw_path_look_ahead[look_ahead_distance] = yaw_path_look_ahead

            yaw_rate_diff_look_ahead = yaw_rate_look_ahead - self._last_yaw_rate_look_ahead[look_ahead_distance] if self._last_yaw_rate_look_ahead[look_ahead_distance] is not None else 0.0
            self._last_yaw_rate_look_ahead[look_ahead_distance] = yaw_rate_look_ahead

            p_crosstrack_ref = p_look_ahead_extrapolation
            crosstrack_error_look_ahead = path.get_lateral_distance(
                p_crosstrack_ref,
                linear_projection=False
            )
            yaw_cross_track = np.arctan2(p_crosstrack_ref[1] - p_proj_front[1], p_crosstrack_ref[0] - p_proj_front[0])
            yaw_path2ct = relative_orientation(yaw_cross_track, yaw_path_front)
            if yaw_path2ct > 0:
                crosstrack_error_look_ahead = abs(crosstrack_error_look_ahead)
            else:
                crosstrack_error_look_ahead = - abs(crosstrack_error_look_ahead)

            look_ahead_observations.append(LookAheadPathObservation(
                crosstrack_error=crosstrack_error_look_ahead,
                yaw_diff=yaw_diff_look_ahead,
                yaw_rate_diff=yaw_rate_diff_look_ahead / ego_vehicle_simulation.dt,
                yaw_rate_path=yaw_rate_path_look_ahead
            ))

        obs = PathObservation(
            yaw_diff_front=yaw_diff_front,
            yaw_diff_center=yaw_diff_center,
            yaw_rate_diff_front=yaw_rate_diff_front / ego_vehicle_simulation.dt,
            crosstrack_error=crosstrack_error, # don't change order ... PathFollowingRewardComputer dependence
            yaw_rate_front=yaw_rate_front,
            yaw_rate_path_front=yaw_rate_path_front,
            look_ahead_observations=look_ahead_observations
        )

        # projected_yaw_front = path.get_projected_direction(p_front)
        # print(f"{yaw_path_front=:.3f}, {projected_yaw_front=:.3f}, {yaw=:.3f}, {yaw_diff_front=:.3f}")

        return obs

    def reset(
        self,
    ) -> None:

        self._last_arclength = None
        self._last_yaw_diff_look_ahead = None
        self._last_yaw_rate_look_ahead  = None
        self._last_yaw_path_look_ahead = None
        self._last_yaw_diff_front = None
        self._last_yaw_rate_front  = None
        self._last_yaw_path_front = None
        self._last_steering_angle = None
