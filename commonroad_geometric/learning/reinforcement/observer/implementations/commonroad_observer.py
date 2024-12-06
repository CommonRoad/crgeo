from __future__ import annotations

import os
from typing import Dict, Union, Optional, List
import yaml
import numpy as np
import gymnasium
from gymnasium import spaces

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.observer.implementations.commonroad_observer_components.lane_marker_observation import LaneMarkerObservation
from commonroad_geometric.learning.reinforcement.observer.implementations.commonroad_observer_components.surrounding_vehicle_detection import SurroundingVehicleDetection, DetectionMode

class CommonRoadObserver(SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    A clean implementation of a CommonRoad observer that provides observations of the traffic environment
    for RL agents.
    """
    def __init__(self):
        """
        
        Args:
            configs: Optional configuration dictionary. If None, loads from default yaml.
        """
        super().__init__()
        
        # Store whether to flatten observation
        self._flatten_observation = True

    def setup(self, dummy_data: CommonRoadData) -> gymnasium.Space:
        """
        Set up the observation space based on enabled features.
        
        Args:
            dummy_data: Example CommonRoadData instance (not used in current implementation)
            
        Returns:
            gymnasium.Space: The observation space
        """
        observation_spaces = {}
        
        observation_spaces["v_ego"] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_spaces["a_ego"] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_spaces["steering_angle"] = spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        observation_spaces["global_turn_rate"] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)     
        observation_spaces['left_marker_distance'] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_spaces['right_marker_distance'] = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_spaces['surrounding_zones'] = spaces.Box(-np.inf, np.inf, (6*4,), dtype=np.float32)

        if self._flatten_observation:
            # Combine all spaces into a single Box space
            total_size = sum(space.shape[0] for space in observation_spaces.values())
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32
            )
        else:
            return spaces.Dict(observation_spaces)

    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate an observation based on current state.
        
        Args:
            data: Current CommonRoadData instance
            ego_vehicle_simulation: Current ego vehicle simulation instance
            
        Returns:
            Union[Dict[str, np.ndarray], np.ndarray]: Either a flattened observation array or
                                                     a dictionary of observation arrays
        """
        observation_dict = {}
        
        # Get ego vehicle state
        ego_state = ego_vehicle_simulation.ego_vehicle.state
        
        # Ego vehicle observations
        observation_dict["v_ego"] = np.array([ego_state.velocity], dtype=np.float32)
        observation_dict["a_ego"] = np.array([ego_state.acceleration], dtype=np.float32)
        observation_dict["steering_angle"] = np.array([ego_state.steering_angle], dtype=np.float32)
        observation_dict["global_turn_rate"] = np.array([ego_state.yaw_rate], dtype=np.float32)
            
        lane_marker_observation = LaneMarkerObservation(
            vehicle_state=ego_state,
            lanelet_network=ego_vehicle_simulation.current_scenario.lanelet_network,
        )
        left_dist, right_dist = lane_marker_observation.get_lane_marker_distances(ego_vehicle_simulation.current_lanelets[0]) if ego_vehicle_simulation.current_lanelets else (0.0, 0.0)
        observation_dict["left_marker_distance"] = np.array([left_dist], dtype=np.float32)
        observation_dict["right_marker_distance"] = np.array([right_dist], dtype=np.float32)

        surrounding_vehicle_detection = SurroundingVehicleDetection(
            vehicle_state=ego_state,
            lanelet_network=ego_vehicle_simulation.current_scenario.lanelet_network,
            detection_config=dict() # use default config
        )
        if ego_vehicle_simulation.current_lanelets:
            surrounding_info = surrounding_vehicle_detection.detect_surrounding_vehicles(
                ego_lanelet=ego_vehicle_simulation.current_lanelets[0],
                mode=DetectionMode.LANE_BASED,
                obstacles=ego_vehicle_simulation.current_non_ego_obstacles,
                time_step=ego_vehicle_simulation.current_time_step
            )
        else:
            surrounding_info = None

        
        # Define a function to process each zone
        def process_zone(zone):
            if zone is None:
                # Placeholder for missing zone: [x, y, velocity, distance]
                return [0.0, 0.0, 0.0, 0.0]
            else:
                # Extract data from zone
                relative_position = zone['relative_position']
                relative_velocity = zone['relative_velocity']
                distance = zone['distance']
                return [relative_position[0], relative_position[1], relative_velocity, distance]

        if surrounding_info is None:
            # Placeholder for missing zones
            zones = [[0.0, 0.0, 0.0, 0.0]] * 6
            observation_dict["surrounding_zones"] = np.array(zones, dtype=np.float32)
        else:
            # Process all zones in the dictionary
            zones = [
                process_zone(surrounding_info['same_lane_front']),
                process_zone(surrounding_info['same_lane_rear']),
                process_zone(surrounding_info['left_lane_front']),
                process_zone(surrounding_info['left_lane_rear']),
                process_zone(surrounding_info['right_lane_front']),
                process_zone(surrounding_info['right_lane_rear'])
            ]
        observation_dict["surrounding_zones"] = np.array(zones, dtype=np.float32)

        
        if self._flatten_observation:
            # Convert dict to flat array
            return np.concatenate([v.flatten() for v in observation_dict.values()])
        else:
            return observation_dict

    def reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        scenario=None,
        planning_problem=None,
        reset_config=None,
        benchmark_id=""
    ):
        """
        Reset the observer with new scenario and config.
        
        Args:
            ego_vehicle_simulation: Current ego vehicle simulation instance
            scenario: The CommonRoad scenario
            planning_problem: The planning problem to solve
            reset_config: Configuration for resetting
            benchmark_id: ID of the benchmark scenario
        """
        # TODO: Store scenario, planning problem and other necessary info
        # Reset any internal state if needed
        pass

    def debug_dict(
        self,
        observation: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> Dict[str, str]:
        """
        Provide debug information about the observation.
        
        Args:
            observation: The observation to debug
            
        Returns:
            Dict[str, str]: Dictionary of debug strings
        """
        debug_info = {}
        
        if isinstance(observation, dict):
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    debug_info[key] = f"Shape: {value.shape}, Mean: {value.mean():.3f}"
        else:
            debug_info["observation"] = f"Shape: {observation.shape}, Mean: {observation.mean():.3f}"
            
        return debug_info