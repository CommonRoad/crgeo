import numpy as np

class LaneMarkerObservation:
    """Handles observations related to lane markers and boundaries"""

    def __init__(self, vehicle_state, lanelet_network):
        self.vehicle_state = vehicle_state
        self.lanelet_network = lanelet_network
        
    def get_lane_marker_distances(self, ego_lanelet):
        """
        Calculate distances to left and right lane markers from ego vehicle position
        Returns distances in meters
        """
        # Get ego vehicle position
        ego_pos = self.vehicle_state.position
        
        # Get left and right boundaries of current lanelet
        left_boundary = ego_lanelet.left_vertices
        right_boundary = ego_lanelet.right_vertices
        
        # Find closest points on boundaries
        left_dist = self._point_to_polyline_distance(ego_pos, left_boundary)
        right_dist = self._point_to_polyline_distance(ego_pos, right_boundary)
        
        return left_dist, right_dist

    def _point_to_polyline_distance(self, point, polyline):
        """Calculate minimum distance from point to polyline"""
        min_dist = float('inf')
        for i in range(len(polyline)-1):
            # Get line segment
            p1 = polyline[i]
            p2 = polyline[i+1]
            
            # Calculate distance to line segment
            dist = self._point_to_line_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
            
        return min_dist
        
    def _point_to_line_segment_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        # Convert to numpy arrays
        p = np.array(point)
        a = np.array(line_start) 
        b = np.array(line_end)
        
        # Calculate projection
        ab = b - a
        ap = p - a
        
        # Calculate normalized distance along line segment
        t = np.dot(ap, ab) / np.dot(ab, ab)
        
        if t < 0:
            # Point projects before line segment
            return np.linalg.norm(p - a)
        elif t > 1:
            # Point projects after line segment
            return np.linalg.norm(p - b)
        else:
            # Point projects onto line segment
            projection = a + t * ab
            return np.linalg.norm(p - projection)