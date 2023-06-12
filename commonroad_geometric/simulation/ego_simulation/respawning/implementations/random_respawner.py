from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union

import networkx as nx
import numpy as np
from commonroad.scenario.trajectory import State

from commonroad_geometric.common.io_extensions.lanelet_network import lanelet_orientation_at_position, map_out_lanelets_to_intersections
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.types import Unlimited
from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import BaseRespawner, BaseRespawnerOptions, RespawnerSetupFailure, T_Respawn_Tuple

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

logger = logging.getLogger(__name__)


@dataclass
class RandomRespawnerOptions(BaseRespawnerOptions):
    random_init_arclength: bool = True
    random_goal_arclength: bool = True
    random_start_timestep: bool = True
    only_intersections: bool = False
    route_length: Optional[Union[int, Tuple[int, int]]] = (3, 10)
    init_speed: float = 4.0
    min_goal_distance: Optional[float] = 100.0
    max_goal_distance: Optional[float] = 200.0
    min_remaining_distance: Optional[float] = 45.0
    max_attempts_outer: int = 50
    min_vehicle_distance: Optional[float] = 16.0
    min_vehicle_speed: Optional[float] = 1.5
    min_vehicles_route: Optional[int] = 2
    max_attempts_inner: int = 5


class RandomRespawner(BaseRespawner):
    """
    Respawns ego vehicle at randomly sampled location.
    Safe (i.e. collision-free) initial positions are ensured by a trial-and-error approach.
    """
    # TODO: cleanup entire class, very hacky and messy implementation
    def __init__(
        self,
        options: Optional[RandomRespawnerOptions] = None
    ) -> None:
        options = options or RandomRespawnerOptions()
        self._options: RandomRespawnerOptions = options
        super().__init__(options=options)

    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Respawn_Tuple:
        # assert ego_vehicle_simulation.simulation.current_time_step == 0

        if self._options.random_start_timestep and ego_vehicle_simulation.simulation.final_time_step is not Unlimited:
            start_step_offset = self.rng.randint(0, int(ego_vehicle_simulation.simulation.final_time_step) // 2)
            ego_vehicle_simulation._simulation = ego_vehicle_simulation.simulation( # TODO: Typing
                from_time_step=start_step_offset,
                ego_vehicle=ego_vehicle_simulation.ego_vehicle,
                force=True
            )
            assert ego_vehicle_simulation.current_time_step == start_step_offset

        routes = ego_vehicle_simulation.simulation.routes
        lanelet_network = ego_vehicle_simulation.simulation.lanelet_network

        lanelet_graph = ego_vehicle_simulation.simulation.lanelet_graph
        traffic_flow_graph = lanelet_graph.get_traffic_flow_graph()
        dfs_continuations = {}

        only_intersections = len(lanelet_network.intersections) > 0 and self._options.only_intersections
        if only_intersections:
            goal_lanelet_candidates = set.union(
                #set(lanelet_network.map_inc_lanelets_to_intersections.keys())
                set(map_out_lanelets_to_intersections(lanelet_network).keys()),
                #set(map_successor_lanelets_to_intersections(lanelet_network).keys())
            )

        attempts_outer: int = 0
        while attempts_outer < self._options.max_attempts_outer:
            success = False

            if isinstance(self._options.route_length, int):
                route_length = self._options.route_length
            else:
                route_length = self.rng.randint(self._options.route_length[0], self._options.route_length[1])

            route_start_lanelet_id = self.rng.choice(list(routes.keys()))
            route_end_lanelet_id = self.rng.choice(list(routes[route_start_lanelet_id].keys()))
            route = routes[route_start_lanelet_id][route_end_lanelet_id]
            
            if only_intersections:
                route_end_index = None
                for candidate in goal_lanelet_candidates:
                    if candidate in route:
                        route_end_index = route.index(candidate) + 1
                if route_end_index is None:
                    attempts_outer += 1
                    continue

                route_start_index = max(0, route_end_index - route_length)
                try:
                    start_lanelet_id = route[route_start_index]
                    goal_lanelet_id = route[route_end_index]
                except IndexError:
                    attempts_outer += 1
                    continue
            else:
                route_start_index = self.rng.choice(list(range(len(route))))
                start_lanelet_id = route[route_start_index]
                route_end_index = min(len(route) - 1, route_start_index + route_length)
                goal_lanelet_id = route[route_end_index]

            route_subset = route[route_start_index:route_end_index+1]
            route_subset_distance = sum((ego_vehicle_simulation.simulation.find_lanelet_by_id(lid).distance[-1] for lid in route_subset))
            if self._options.min_goal_distance is not None and route_subset_distance < self._options.min_goal_distance:
                attempts_outer += 1
                continue
            if self._options.max_goal_distance is not None and route_subset_distance > self._options.max_goal_distance:
                attempts_outer += 1
                continue

            route_vehicle_ids = set.union(*(set(ego_vehicle_simulation.simulation.get_obstacles_on_lanelet(lid)) for lid in route_subset))
            start_lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(start_lanelet_id)
            goal_lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(goal_lanelet_id)

            if self._options.random_goal_arclength:
                goal_arclength = min(goal_lanelet.distance[-1] - self._options.goal_region_length / 2, self.rng.random() * goal_lanelet.distance[-1])
            else:
                goal_arclength = goal_lanelet.distance[-1] - self._options.goal_region_length / 2
            goal_arclength = max(0.0, min(goal_arclength, goal_lanelet.distance[-1]))

            if self._options.min_remaining_distance is not None:
                if goal_lanelet_id not in dfs_continuations:
                    dfs_continuation = list(nx.dfs_preorder_nodes(traffic_flow_graph, source=goal_lanelet_id))
                    dfs_continuations[goal_lanelet_id] = dfs_continuation
                else:
                    dfs_continuation = dfs_continuations[goal_lanelet_id]
                total_remaining = sum((
                    ego_vehicle_simulation.simulation.find_lanelet_by_id(successor).distance[-1]
                ) for successor in dfs_continuation) - goal_arclength
                if total_remaining < self._options.min_remaining_distance:
                    attempts_outer += 1
                    continue

            self._goal_position = goal_lanelet.interpolate_position(goal_arclength)[0]

            attempts_inner: int = -1
            while attempts_inner < self._options.max_attempts_inner:
                attempts_inner += 1
                start_offset = 10.0 if not start_lanelet.predecessor else 0.0
                final_offset = 10.0 if not goal_lanelet.successor else 0.0
                if self._options.random_init_arclength:
                    start_arclength = start_offset + (start_lanelet.distance[-1] - final_offset) * self.rng.random()
                else:
                    start_arclength = start_offset
                start_arclength = max(0.0, min(start_arclength, start_lanelet.distance[-1]))
                start_position = start_lanelet.interpolate_position(start_arclength)[0]
                goal_distance = route_subset_distance - start_arclength - (goal_lanelet.distance[-1] - goal_arclength)

                if goal_distance < 0 or self._options.min_goal_distance is not None and goal_distance < self._options.min_goal_distance:
                    continue
                if self._options.max_goal_distance is not None and goal_distance > self._options.max_goal_distance:
                    continue
                
                if self._options.min_vehicle_distance is not None or self._options.min_vehicle_speed is not None or self._options.min_vehicles_route is not None:
                    valid_lanelet_environment = True
                    threshold_vehicles = min(len(ego_vehicle_simulation.simulation.current_scenario.obstacles), self._options.min_vehicles_route)
                    if self._options.min_vehicles_route is not None and len(route_vehicle_ids) < threshold_vehicles:
                        continue
                    for obstacle_id in route_vehicle_ids:
                        obstacle = ego_vehicle_simulation.simulation.current_scenario._dynamic_obstacles[obstacle_id]
                        obstacle_state = state_at_time(obstacle, ego_vehicle_simulation.current_time_step, assume_valid=True)
                        if self._options.min_vehicle_speed is not None and obstacle_state.velocity < self._options.min_vehicle_speed:
                            valid_lanelet_environment = False
                            break
                        obstacle_distance = np.linalg.norm(start_position - obstacle_state.position)
                        if obstacle_distance < self._options.min_vehicle_distance:
                            valid_lanelet_environment = False
                            break
                    if not valid_lanelet_environment:
                        continue

                success = True
                break

            if success:
                logger.debug(f"{type(self).__name__} suggests route {route}")
                break

            attempts_outer += 1

        if not success and self._options.throw_on_failure:
            raise RespawnerSetupFailure(f"Max respawn attempts reached ({attempts_outer})")

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
