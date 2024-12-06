import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

class DetectionMode(Enum):
    LANE_BASED = "lane_based"
    LIDAR_BASED = "lidar_based"

class SurroundingVehicleDetection:
    """Handles detection of surrounding vehicles using either lane-based or lidar-based approaches"""
    
    def __init__(self, vehicle_state, lanelet_network, detection_config):
        self.vehicle_state = vehicle_state
        self.lanelet_network = lanelet_network
        self.config = detection_config
        
        # Configuration parameters
        self.lane_detection_range_front = detection_config.get('lane_range_front', 100.0)
        self.lane_detection_range_rear = detection_config.get('lane_range_rear', 50.0)
        self.lidar_range = detection_config.get('lidar_range', 100.0)
        self.lidar_angles = detection_config.get('lidar_angles', 
                                               np.linspace(-np.pi, np.pi, 72))  # 5-degree resolution
        
    def detect_surrounding_vehicles(self, 
                                  ego_lanelet,
                                  mode: DetectionMode,
                                  obstacles: List,
                                  time_step: int) -> Dict:
        """
        Detect surrounding vehicles using specified detection mode
        Returns dict with detected vehicle information
        """
        if mode == DetectionMode.LANE_BASED:
            return self._lane_based_detection(ego_lanelet, obstacles, time_step)
        else:
            return self._lidar_based_detection(obstacles, time_step)
            
    def _lane_based_detection(self, ego_lanelet, obstacles: List, time_step: int) -> Dict:
        """
        Detect vehicles in ego lane and adjacent lanes
        Returns dict with nearest vehicles in each relevant position
        """
        ego_pos = self.vehicle_state.position
        ego_vel = self.vehicle_state.velocity
        
        # Initialize detection zones
        zones = {
            'same_lane_front': None,
            'same_lane_rear': None,
            'left_lane_front': None,
            'left_lane_rear': None,
            'right_lane_front': None,
            'right_lane_rear': None
        }
        
        # Get adjacent lanelets
        adj_left_id = ego_lanelet.adj_left
        adj_right_id = ego_lanelet.adj_right
        
        for obstacle in obstacles:
            # Skip if obstacle is ego vehicle
            if obstacle.obstacle_id == -1:
                continue
                
            obs_state = obstacle.state_at_time(time_step)
            if obs_state is None:
                continue
                
            obs_pos = obs_state.position
            obs_vel = obs_state.velocity
            
            # Calculate relative position and velocity
            rel_pos = obs_pos - ego_pos
            rel_vel = obs_vel - ego_vel
            
            # Get obstacle's lanelet
            obs_lanelet_ids = self.lanelet_network.find_lanelet_by_position([obs_pos])[0]
            
            # Check which zone the obstacle belongs to
            for obs_lanelet_id in obs_lanelet_ids:
                if obs_lanelet_id == ego_lanelet.lanelet_id:
                    if rel_pos[0] > 0 and rel_pos[0] < self.lane_detection_range_front:
                        zones['same_lane_front'] = self._update_vehicle_info(
                            zones['same_lane_front'], rel_pos, rel_vel, obstacle)
                    elif rel_pos[0] < 0 and abs(rel_pos[0]) < self.lane_detection_range_rear:
                        zones['same_lane_rear'] = self._update_vehicle_info(
                            zones['same_lane_rear'], rel_pos, rel_vel, obstacle)
                            
                elif obs_lanelet_id == adj_left_id:
                    if rel_pos[0] > 0:
                        zones['left_lane_front'] = self._update_vehicle_info(
                            zones['left_lane_front'], rel_pos, rel_vel, obstacle)
                    else:
                        zones['left_lane_rear'] = self._update_vehicle_info(
                            zones['left_lane_rear'], rel_pos, rel_vel, obstacle)
                            
                elif obs_lanelet_id == adj_right_id:
                    if rel_pos[0] > 0:
                        zones['right_lane_front'] = self._update_vehicle_info(
                            zones['right_lane_front'], rel_pos, rel_vel, obstacle)
                    else:
                        zones['right_lane_rear'] = self._update_vehicle_info(
                            zones['right_lane_rear'], rel_pos, rel_vel, obstacle)
                        
        return zones
        
    def _lidar_based_detection(self, obstacles: List, time_step: int) -> Dict:
        """
        Simulate lidar-like detection of surrounding vehicles
        Returns dict with detected obstacles for each angle
        """
        ego_pos = self.vehicle_state.position
        ego_orientation = self.vehicle_state.orientation
        
        # Initialize detection results
        detections = {angle: None for angle in self.lidar_angles}
        
        for obstacle in obstacles:
            # Skip if obstacle is ego vehicle
            if obstacle.obstacle_id == -1:
                continue
                
            obs_state = obstacle.state_at_time(time_step)
            if obs_state is None:
                continue
                
            obs_pos = obs_state.position
            
            # Calculate relative position
            rel_pos = obs_pos - ego_pos
            distance = np.linalg.norm(rel_pos)
            
            # Skip if outside range
            if distance > self.lidar_range:
                continue
                
            # Calculate angle to obstacle
            angle = np.arctan2(rel_pos[1], rel_pos[0]) - ego_orientation
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            
            # Find closest angle in lidar_angles
            closest_angle_idx = np.argmin(np.abs(self.lidar_angles - angle))
            closest_angle = self.lidar_angles[closest_angle_idx]
            
            # Update detection if closer than previous detection at this angle
            if detections[closest_angle] is None or distance < detections[closest_angle]['distance']:
                detections[closest_angle] = {
                    'obstacle_id': obstacle.obstacle_id,
                    'distance': distance,
                    'angle': angle,
                    'state': obs_state
                }
                
        return detections
        
    def _update_vehicle_info(self, 
                            current_info: Optional[Dict], 
                            rel_pos: np.ndarray,
                            rel_vel: float, 
                            obstacle) -> Dict:
        """Helper to update vehicle information based on distance"""
        new_info = {
            'relative_position': rel_pos,
            'relative_velocity': rel_vel,
            'obstacle_id': obstacle.obstacle_id,
            'distance': np.linalg.norm(rel_pos)
        }
        
        if current_info is None or new_info['distance'] < current_info['distance']:
            return new_info
            
        return current_info