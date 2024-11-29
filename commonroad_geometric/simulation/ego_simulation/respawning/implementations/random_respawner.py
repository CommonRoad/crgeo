from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple, Union

import networkx as nx
import numpy as np
from commonroad.scenario.state import State, InitialState
from scipy.spatial import cKDTree

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
    route_length: Optional[Union[int, Tuple[int, int]]] = (3, 15)
    init_speed: Union[str, float] = 10.0
    random_speed_range: Tuple[float, float] = (0.0, 10.0)
    normal_speed_params: Tuple[float, float] = (10.0, 2.0)  # (mean, std_dev)
    init_steering_angle: Union[str, float] = 0.0
    random_steering_angle_range: Tuple[float, float] = (-0.1, 0.1)
    normal_steering_angle_params: Tuple[float, float] = (0.0, 0.05)  # (mean, std_dev)
    init_orientation_noise: float = 0.0
    init_position_noise: float = 0.0
    min_init_arclength: Optional[float] = 0.0
    min_goal_distance: Optional[float] = 100.0
    min_goal_distance_l2: Optional[float] = 100.0
    max_goal_distance: Optional[float] = 200.0
    max_goal_distance_l2: Optional[float] = 200.0
    min_remaining_distance: Optional[float] = None
    max_attempts_outer: int = 50
    min_vehicle_distance: Optional[float] = 12.0
    future_timestep_count: int = 5
    min_vehicle_speed: Optional[float] = None
    min_vehicles_route: Optional[int] = None
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
        if isinstance(options, dict):
            options = RandomRespawnerOptions(**options)
        self._options: RandomRespawnerOptions = options
        super().__init__(options=options)

    def _check_future_vehicle_proximity(self, ego_vehicle_simulation: EgoVehicleSimulation, start_position: np.ndarray) -> bool:
        """
        Checks if any vehicles will pass within a certain distance from the ego vehicle within the next N timesteps.
        Returns True if the ego vehicle's position is valid (i.e., no nearby vehicles in the future).
        """
        future_timestep_count = self._options.future_timestep_count
        min_future_vehicle_distance = self._options.min_vehicle_distance

        current_time_step = ego_vehicle_simulation.current_time_step
        dynamic_obstacles = ego_vehicle_simulation.simulation.current_scenario.dynamic_obstacles

        for future_step in range(1, future_timestep_count + 1):
            future_timestep = current_time_step + future_step

            obstacle_positions = []
            for obstacle in dynamic_obstacles:
                obstacle_state = state_at_time(obstacle, future_timestep, assume_valid=False)
                if obstacle_state is not None:
                    obstacle_positions.append(obstacle_state.position)

            if not obstacle_positions:
                continue  # No obstacles at this timestep

            # Build KD-tree for current future timestep
            obstacle_positions = np.array(obstacle_positions)
            kd_tree = cKDTree(obstacle_positions)

            # Query the KD-tree to find obstacles within the minimum distance
            indices = kd_tree.query_ball_point(start_position, r=min_future_vehicle_distance)
            if indices:
                return False  # Obstacle found within the minimum distance

        return True  # No obstacles within the minimum distance in future timesteps

    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Respawn_Tuple:
        # assert ego_vehicle_simulation.simulation.current_time_step == 0

        if self._options.random_start_timestep and ego_vehicle_simulation.simulation.final_time_step is not Unlimited:
            start_step_offset = self.rng.randint(int(ego_vehicle_simulation.simulation.initial_time_step), int(ego_vehicle_simulation.simulation.final_time_step) // 4)
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
                start_offset = self._options.min_init_arclength if not start_lanelet.predecessor else 1.0
                final_offset = 10.0 if not goal_lanelet.successor else 1.0
                if self._options.random_init_arclength:
                    start_arclength = start_offset + (start_lanelet.distance[-1] - final_offset) * self.rng.random()
                else:
                    start_arclength = start_offset
                start_arclength = max(0.0, min(start_arclength, start_lanelet.distance[-1]))
                if start_arclength < self._options.min_init_arclength:
                    continue
                start_position = start_lanelet.interpolate_position(start_arclength)[0]
                goal_distance = route_subset_distance - start_arclength - (goal_lanelet.distance[-1] - goal_arclength)
                goal_distance_l2 = np.linalg.norm(self._goal_position - start_position)

                if goal_distance < 0 or self._options.min_goal_distance is not None and goal_distance < self._options.min_goal_distance:
                    continue # Invalid because too close to goal
                if self._options.min_goal_distance_l2 is not None and goal_distance_l2 < self._options.min_goal_distance_l2:
                    continue # Invalid because too close to goal
                if self._options.max_goal_distance is not None and goal_distance > self._options.max_goal_distance:
                    continue
                if self._options.max_goal_distance_l2 is not None and goal_distance_l2 > self._options.max_goal_distance_l2:
                    continue
                if goal_lanelet_id == start_lanelet_id and goal_arclength < start_arclength:
                    continue
                
                if self._options.min_vehicle_distance is not None or self._options.min_vehicle_speed is not None or self._options.min_vehicles_route is not None:
                    valid_lanelet_environment = True

                    if self._options.min_vehicles_route is not None:
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

                    if valid_lanelet_environment:
                        valid_lanelet_environment = self._check_future_vehicle_proximity(ego_vehicle_simulation, start_position)

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
        
        if self._options.init_speed == "auto":
            vehicle_states = [state_at_time(obstacle, ego_vehicle_simulation.current_time_step) for obstacle in ego_vehicle_simulation.simulation.current_scenario.dynamic_obstacles]
            vehicle_states = [state for state in vehicle_states if state is not None]
            vehicle_speeds = [state.velocity for state in vehicle_states if state is not None]
            vehicle_ego_distances = [np.linalg.norm(start_position - state.position) for state in vehicle_states]
            if len(vehicle_ego_distances) > 0:
                closest_vehicle_index = min(range(len(vehicle_ego_distances)), key=lambda i: vehicle_ego_distances[i])
                if vehicle_ego_distances[closest_vehicle_index] < 50.0:
                    init_speed = vehicle_speeds[closest_vehicle_index]
                else:
                    init_speed = 5.0
            else:
                # fallback if no vehicles in scenario
                init_speed = 5.0
        elif self._options.init_speed == "uniform_random":
            # Select a speed randomly within the specified range
            init_speed = self.rng.uniform(*self._options.random_speed_range)
        elif self._options.init_speed == "normal_random":
            # Select a speed using a normal distribution
            mean, std_dev = self._options.normal_speed_params
            init_speed = self.rng.gauss(mean, std_dev)
            # Ensure the speed is within the specified range
            init_speed = max(self._options.random_speed_range[0], min(self._options.random_speed_range[1], init_speed))
        else:
            init_speed = self._options.init_speed

        if self._options.init_steering_angle == "uniform_random":
            init_steering_angle = self.rng.uniform(*self._options.random_steering_angle_range)
        elif self._options.init_steering_angle == "normal_random":
            # Select a steering angle using a normal distribution
            mean, std_dev = self._options.normal_steering_angle_params
            init_steering_angle = self.rng.gauss(mean, std_dev)
            # Ensure the steering angle is within the specified range
            init_steering_angle = max(self._options.random_steering_angle_range[0], min(self._options.random_steering_angle_range[1], init_steering_angle))
        else:
            init_steering_angle = self._options.init_steering_angle

        start_orientation = lanelet_orientation_at_position(start_lanelet, start_position)

        if self._options.init_orientation_noise > 0.0:
            start_orientation += self.rng.gauss(0, self._options.init_orientation_noise)
        if self._options.init_position_noise > 0.0:
            start_position[0] += self.rng.gauss(0, self._options.init_position_noise)
            start_position[1] += self.rng.gauss(0, self._options.init_position_noise)

        initial_state = InitialState(
            position=start_position,
            velocity=init_speed,
            orientation=start_orientation,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=ego_vehicle_simulation.current_time_step if ego_vehicle_simulation.current_time_step is not None else ego_vehicle_simulation.initial_time_step
        )


        initial_state.steering_angle = init_steering_angle

        return initial_state, self._goal_position, goal_lanelet
