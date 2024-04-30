from __future__ import annotations
from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
import numpy as np
from typing import Dict, List, Optional
from gymnasium import spaces
import logging
from copy import deepcopy
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.distance_to_road_boundary_feature_computer import DistanceToRoadBoundariesFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle.ego.goal_alignment_feature_computer import GoalAlignmentComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import VFeatureParams
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.utils.path_observer import PathObserver, flatten_path_observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver

logger = logging.getLogger(__name__)


class EgoEnhancedGraphObserver(BaseObserver):
    """
    Flattens a CommonRoadData graph object into a dictionary of fixed-sized, padded numpy arrays.
    """
    def __init__(
        self, 
        data_padding_size: Optional[int] = None,
        global_features_include: Optional[List[str]] = None,
        include_path_observations: bool = True,
        look_ahead_distances: Optional[list[float]] = None,
        include_graph_observations: bool = True
    ) -> None:
        self.include_graph_observations = include_graph_observations
        if look_ahead_distances is None:
            look_ahead_distances = [2.5, 5.0, 10.0, 15.0]
        self.look_ahead_distances = look_ahead_distances
        if include_graph_observations:
            self.graph_observer = FlattenedGraphObserver(
                data_padding_size=data_padding_size,
                global_features_include=global_features_include
            )
        self.include_path_observations = include_path_observations
        self.path_observer = PathObserver(look_ahead_distances=look_ahead_distances)
        self.goal_alignment_computer = GoalAlignmentComputer(
            include_lane_changes_required=True
        )
        self.distance_to_road_boundary_computer = DistanceToRoadBoundariesFeatureComputer()
        super().__init__()

    def setup(self, dummy_data: CommonRoadData) -> spaces.Space:
        if self.include_graph_observations:
            graph_observation_space = self.graph_observer.setup(dummy_data)
            # TODO AUTO DIMS
            observation_space = deepcopy(graph_observation_space)
        else:
            observation_space = spaces.Dict({})
        if self.include_path_observations:
            observation_space["path_observation"] = spaces.Box(-np.inf, np.inf, (6 + 4*len(self.look_ahead_distances), ), dtype=np.float32)
        observation_space["ego_observation"] = spaces.Box(-np.inf, np.inf, (5, ), dtype=np.float32) # TODO auto size
        observation_space["goal_observation"] = spaces.Box(-np.inf, np.inf, (6, ), dtype=np.float32) # TODO auto size
        observation_space["road_observation"] = spaces.Box(-np.inf, np.inf, (3, ), dtype=np.float32) # TODO auto size

        return observation_space

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:

        ego_state = ego_vehicle_simulation.ego_vehicle.state
        ego_parameters = ego_vehicle_simulation.ego_vehicle.parameters

        if self.include_graph_observations:
            obs = self.graph_observer.observe(
                data=data,
                ego_vehicle_simulation=ego_vehicle_simulation
            )
        else:
            obs = {}

        if self.include_path_observations:
            path_observation = self.path_observer.observe(
                ego_vehicle_simulation=ego_vehicle_simulation
            )
            # print(d, path_observation)
            obs["path_observation"] = flatten_path_observation(path_observation)

        ego_v_params = VFeatureParams(
            dt=ego_vehicle_simulation.dt,
            time_step=ego_vehicle_simulation.current_time_step,
            obstacle=ego_vehicle_simulation.ego_vehicle.as_dynamic_obstacle,
            state=ego_state,
            is_ego_vehicle=True,
            ego_state=ego_state,
            ego_route=ego_vehicle_simulation.ego_route
        )
        goal_alignment_dict = self.goal_alignment_computer(
            params=ego_v_params,
            simulation=ego_vehicle_simulation.simulation
        )
        obs["goal_observation"] = np.array(list(goal_alignment_dict.values()))

        distance_to_road_boundary_dict = self.distance_to_road_boundary_computer(
            params=ego_v_params,
            simulation=ego_vehicle_simulation.simulation
        )
        obs["road_observation"] = np.array(list(distance_to_road_boundary_dict.values()))

        rel_velocity = ego_state.velocity / ego_parameters.longitudinal.v_max
        rel_acceleration = (ego_state.acceleration if ego_state.acceleration is not None else 0.0) / ego_parameters.longitudinal.a_max
        rel_steering_angle = ego_state.steering_angle / ego_parameters.steering.v_max
        yaw_rate = ego_state.yaw_rate / ego_parameters.steering.max
        reverse = 0 if ego_state.velocity >= 0.0 else 1

        obs["ego_observation"] = np.array([
            rel_velocity,
            rel_acceleration,
            rel_steering_angle,
            yaw_rate,
            reverse
        ])

        return obs

    def reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ):
        if self.include_path_observations:
            self.path_observer.reset()
        self.goal_alignment_computer.reset(ego_vehicle_simulation.simulation)
        self.distance_to_road_boundary_computer.reset(ego_vehicle_simulation.simulation)

    def debug_dict(
        self,
        observation: T_Observation
    ) -> Dict[str, str]:
        debug_dict = {
            "ego": numpy_prettify(observation["ego_observation"]),
            "goal": numpy_prettify(observation["goal_observation"]),
            "road": numpy_prettify(observation["road_observation"]),
        }
        if self.include_path_observations:
            debug_dict["path"] =  numpy_prettify(observation["path_observation"])
        return debug_dict