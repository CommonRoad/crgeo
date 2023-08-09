from enum import Enum, unique

@unique
class V_Feature(Enum):
    """
    Vehicle nodes represents traffic participants at a specific time.
    Since our graph should be transformation-invariant,
    we do not include the absolute position and orientation of a vehicle in the node attributes. 
    """
    Velocity = 'velocity'
    Acceleration = 'acceleration'
    Orientation = 'orientation'
    YawRate = 'yaw_rate'
    OrientationVec = 'orientation_vec'
    Length = 'length'
    Width = 'width'
    LaneletArclengthAbs = 'lanelet_arclength_abs' 
    LaneletArclengthRel = 'lanelet_arclength_rel'
    DistLeftBound = 'dist_left_bound'
    DistRightBound = 'dist_right_bound'
    LaneletLateralError = 'lanelet_lateral_error'
    HeadingError = 'heading_error'
    HasAdjLaneLeft = 'has_adj_lane_left'
    HasAdjLaneRight = 'has_adj_lane_right'
    Vertices = 'vertices'
    GoalDistanceLongitudinal = 'goal_distance_long'
    GoalDistanceLateral = 'goal_distance_lat'
    GoalDistance = 'goal_distance'
    LaneChangesRequired = 'lane_changes_required'
    LaneChangeDirectionRequired = 'lane_change_dir_required'
    PosEgoFrame = 'position_ego_frame'
    AngleEgoFrame = 'angle_ego_frame'
    VelocityEgoFrame = 'velocity_ego_frame'
    NumLanaletAssignments = 'num_lanelet_assignments'

@unique
class L_Feature(Enum):
    StartCurvature = 'start_curvature'
    Curvature = 'curvature'
    EndCurvature = 'end_curvature'
    DirectionChange = 'direction_change'
    Orientation = 'lanelet_orientation'
    Length = 'lanelet_length'

@unique
class V2V_Feature(Enum):
    """
    vehicle edges represent interaction between vehicles, allowing the model to learn interaction-aware representations.
    """
    SameLanelet = 'same_lanelet'
    Closeness = 'closeness'
    LogCloseness = 'log_closeness'
    Distance = 'distance'
    RelativePosition = 'rel_position'
    RelativeVelocity = 'rel_velocity'
    RelativeOrientation = 'rel_orientation'
    RelativeAcceleration = 'rel_acceleration'
    DistanceEgo = 'distance_ego'
    RelativePositionEgo = 'rel_position_ego'
    RelativeVelocityEgo = 'rel_velocity_ego'
    RelativeOrientationEgo = 'rel_orientation_ego'
    RelativeAccelerationEgo = 'rel_acceleration_ego'
    TimeToClosest = 'time_to_closest'
    ClosestDistance = 'closest_distance'
    ExpectsCollision = 'expects_collision'
    TimeToCollision = 'time_to_collision'
    TimeToClosestCLT = 'time_to_closest_cl_transform'
    ClosestDistanceCLT = 'closest_distance_cl_transform'
    TimeToCollisionCLT = 'time_to_collision_cl_transform'

@unique
class L2L_Feature(Enum):
    RelativeIntersectAngle = 'relative_intersect_angle'
    RelativeStartAngle = 'relative_start_angle'
    RelativeEndAngle = 'relative_end_angle'
    SourcetArclengthAbs = 'source_arclength_abs'
    SourceArclengthRel = 'source_arclength_rel'
    TargetArclengthAbs = 'target_arclength_abs'
    TargetArclengthRel = 'target_arclength_rel'
    TargetCurvature = 'target_curvature'
    SourceCurvature = 'source_curvature'
    RelativeSourceLength = 'relative_source_length'
    EdgeType = 'lanelet_edge_type'
    Weight = 'weight'
    TrafficFlow = 'traffic_flow'
    RelativeOrientation = 'relative_orientation'
    RelativePosition = 'relative_position'
    Distance = 'distance'

@unique
class V2L_Feature(Enum):
    """
    Each vehicle v is connected to all lanelets which contain the vehicle center at the current time step 
    via bidirectional vehicle-lanelet edges. Bidirectional edges are represented as two separate directed edges, 
    each with identical edge attributes.
    """
    V2LLaneletArclengthAbs = 'v2l_lanelet_arclength_abs'
    V2LLaneletArclengthRel = 'v2l_lanelet_arclength_rel'
    V2LDistLeftBound = 'v2l_dist_left_bound'
    V2LDistRightBound = 'v2l_dist_right_bound'
    V2LLaneletLateralError = 'v2l_lanelet_lateral_error'
    V2LHeadingError = 'v2l_heading_error'

@unique
class L2V_Feature(Enum):
    ...