from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.io_extensions.lanelet_network import lanelet_orientation_at_position
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.types import Unlimited
from commonroad_geometric.simulation.ego_simulation.respawning import BaseRespawner, BaseRespawnerOptions

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

logger = logging.getLogger(__name__)


@dataclass
class InBetweenTrafficRespawnerOptions(BaseRespawnerOptions):
    max_respawn_attempts: int = 10
    random_goal_arclength: bool = True
    random_start_timestep: bool = True
    init_speed: float = 5.0
    min_goal_distance: Optional[float] = 50.0
    max_goal_distance: Optional[float] = None
    max_attempts_outer: int = 20
    start_arclength_offset: float = 5.0
    min_threshold_diff: float = 20.0

class InBetweenTrafficRespawner(BaseRespawner):

    def __init__(
        self,
        options: Optional[InBetweenTrafficRespawner] = None
    ) -> None:
        options = options or InBetweenTrafficRespawnerOptions()
        super().__init__(options=options)
        self._options = options
        assert self._options.start_arclength_offset > 5
        assert self._options.min_threshold_diff > self._options.start_arclength_offset

    # def _setup(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
    #     scenario = ego_vehicle_simulation.current_scenario

    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> Tuple[State, np.ndarray, Lanelet, bool]:
        assert ego_vehicle_simulation.simulation.current_time_step == 0

        if self._options.random_start_timestep and ego_vehicle_simulation.simulation.final_time_step is not Unlimited:
            start_step_offset = random.randint(0, int(ego_vehicle_simulation.simulation.final_time_step) // 2)
        else:
            start_step_offset = 0
        ego_vehicle_simulation._simulation = ego_vehicle_simulation.simulation(
            from_time_step=start_step_offset,
            ego_vehicle=ego_vehicle_simulation.ego_vehicle
        )

        routes = ego_vehicle_simulation.simulation.routes
        lanelet_network = ego_vehicle_simulation.simulation.lanelet_network

        attempts_outer: int = 0
        retained_arclength_diff = []
        start_position = None
        unique_list = list(routes.keys())

        try:
            while attempts_outer < self._options.max_attempts_outer and len(unique_list) > 0:
                has_multiple_vehicles = False
                while not has_multiple_vehicles and len(unique_list) > 0:
                    route_start_lanelet_id = random.choice(unique_list)  
                    obstacles_on_start_lanelet = ego_vehicle_simulation.simulation.get_obstacles_on_lanelet(route_start_lanelet_id)
                    if len(obstacles_on_start_lanelet) > 1:
                        has_multiple_vehicles = True
                        break
                    unique_list.remove(route_start_lanelet_id)

                if not has_multiple_vehicles:
                    break

                goal_lanelet_id = random.choice(list(routes[route_start_lanelet_id].keys()))

                start_lanelet_id = route_start_lanelet_id
                start_lanelet = lanelet_network.find_lanelet_by_id(start_lanelet_id)
                center_polyline = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(start_lanelet_id)

                arcLengths = [center_polyline.get_projected_arclength(state_at_time(
                    ego_vehicle_simulation.simulation.current_scenario.obstacle_by_id(obstacle_id), ego_vehicle_simulation.current_time_step, assume_valid=True).position, 
                    relative=False)  for obstacle_id in obstacles_on_start_lanelet]
                arcLengths.sort()

                arcLength_diffs = [arcLengths[i+1] - arcLengths[i] for i in range(len(arcLengths)-1)]
                max_arc_length_diff = max(arcLength_diffs)

                goal_lanelet = lanelet_network.find_lanelet_by_id(goal_lanelet_id)

                if self._options.random_goal_arclength:
                    goal_arclength = min(goal_lanelet.distance[-1] - self._options.goal_region_length / 2, random.random() * goal_lanelet.distance[-1])
                else:
                    goal_arclength = goal_lanelet.distance[-1] - self._options.goal_region_length / 2
                goal_arclength = max(0.0, min(goal_arclength, goal_lanelet.distance[-1]))
                self._goal_position = goal_lanelet.interpolate_position(goal_arclength)[0]

                if max_arc_length_diff > self._options.min_threshold_diff:
                    filtered_list = list(filter(lambda x: x != None, [i if arcLengths[i+1] - arcLengths[i] > self._options.min_threshold_diff else None for i in range(len(arcLengths)-1)]))
                    start_arclength_idx = random.choice(filtered_list)
                    start_arclength = arcLengths[start_arclength_idx]
                    randlimit = int(arcLength_diffs[start_arclength_idx]-2*(self._options.start_arclength_offset))
                    if randlimit < 0:
                        randlimit = int(self._options.start_arclength_offset/2)
                    start_arclength = start_arclength + self._options.start_arclength_offset + random.randrange(0, randlimit)
                    start_position = start_lanelet.interpolate_position(start_arclength)[0]
                    break
                else:                
                    if max_arc_length_diff >= 10.0:
                        retained_arclength_diff.append({start_lanelet_id: arcLength_diffs})

                unique_list.remove(route_start_lanelet_id)
                attempts_outer+=1
        except Exception as e:
            logger.error(e, exc_info=True)
            return

        if start_position is None:
            if len(retained_arclength_diff) > 0:
                key = random.choice(list(retained_arclength_diff.keys()))
                filtered_list = list(filter(lambda x: x != None, [i if retained_arclength_diff[key][i+1] - retained_arclength_diff[key][i] >= 10 else None for i in range(len(retained_arclength_diff[key])-1)]))
                start_arclength = retained_arclength_diff[key][random.choice(filtered_list)]
                start_arclength = start_arclength + self._options.start_arclength_offset
                start_position = start_lanelet.interpolate_position(start_arclength)[0]

        start_orientation = lanelet_orientation_at_position(start_lanelet, start_position)
        initial_state = State(
            position=start_position,
            steering_angle=0.0,
            velocity=self._options.init_speed,
            orientation=start_orientation,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=ego_vehicle_simulation.current_time_step if ego_vehicle_simulation.current_time_step is not None else ego_vehicle_simulation.initial_time_step
        )
        return initial_state, self._goal_position, goal_lanelet
