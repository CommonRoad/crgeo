from collections import OrderedDict
from typing import Union, Dict, List, Tuple, Set, Optional

import commonroad_dc.pycrcc as pycrcc
import gymnasium
import numpy as np
from commonroad.common.util import make_valid_orientation
from commonroad.geometry.shape import Polygon, Rectangle
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.scenario.state import State, CustomState
from commonroad.scenario.obstacle import Obstacle, SignalState, ObstacleRole, ObstacleType, StaticObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer, ZOrders
from commonroad.visualization.util import LineDataUnits
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from commonroad_rl.tools.trajectory_classification import TrajectoryType, classify_trajectory
from numpy import ndarray
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from shapely.geometry import Point, LineString

from commonroad_rl.gym_commonroad.action.vehicle import Vehicle
from commonroad_rl.gym_commonroad.observation.observation import Observation
from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector
from commonroad_rl.gym_commonroad.utils.navigator import Navigator
from commonroad_rl.gym_commonroad.utils.conflict_zone import ConflictZone


class SurroundingObservation(Observation):
    """
    Adapted from commonroad-rl version
    """

    def __init__(self, configs: Dict):
        # Read configs
        self.observe_is_collision: bool = configs.get("observe_is_collision", False)

        self.observe_vehicle_type: bool = configs.get("observe_vehicle_type", False)
        self.num_vehicle_types: int = 7
        self.observe_vehicle_lights: bool = configs.get("observe_vehicle_lights", False)

        self.fast_distance_calculation: bool = configs.get("fast_distance_calculation", True)

        self.observe_lane_rect_surrounding: bool = configs.get("observe_lane_rect_surrounding", False)
        self.lane_rect_sensor_range_length: float = configs.get("lane_rect_sensor_range_length", 35.0)
        self.lane_rect_sensor_range_width: float = configs.get("lane_rect_sensor_range_width", 7)

        self.observe_lane_circ_surrounding: bool = configs.get("observe_lane_circ_surrounding", True)
        self.lane_circ_sensor_range_radius: float = configs.get("lane_circ_sensor_range_radius", 35.0)

        self.observe_lidar_circle_surrounding: bool = configs.get("observe_lidar_circle_surrounding", False)
        self.lidar_circle_num_beams: int = configs.get("lidar_circle_num_beams", 12)
        self.lidar_sensor_radius: float = configs.get("lidar_sensor_radius", 35.0)
        self.observe_lane_change: bool = configs.get("observe_lane_change", False)

        self.observe_relative_priority: bool = configs.get("observe_relative_priority", False)

        self.observe_intersection_velocities = configs.get("observe_intersection_velocities", False)
        self.observe_intersection_distances = configs.get("observe_intersection_distances", False)
        self.observe_ego_distance_intersection = configs.get("observe_ego_distance_intersection", False)
        self.dummy_dist_intersection:float =  configs.get("dummy_dist_intersection",50.)


        self.max_obs_dist: float = 0.0
        if self.observe_lane_circ_surrounding:
            self.max_obs_dist = self.lane_circ_sensor_range_radius
        elif self.observe_lane_rect_surrounding:
            self.max_obs_dist \
                = np.sqrt((self.lane_rect_sensor_range_length / 2) ** 2 + (self.lane_rect_sensor_range_width / 2) ** 2)
        elif self.observe_lidar_circle_surrounding:
            self.max_obs_dist = self.lidar_sensor_radius

        assert sum([self.observe_lidar_circle_surrounding, self.observe_lane_rect_surrounding,
                    self.observe_lane_circ_surrounding]) <= 1, "Only one kind of surrounding observation can be active!"

        self.reward_safe_distance_coef: float = -1 # hybrid_reward_configs.get("reward_safe_distance_coef")
        

        self._local_ccosy = None
        self._scenario: Scenario = None
        self._current_time_step = None
        self._ego_state = None
        self._last_ego_lanelet_id = None
        self._surrounding_area = None
        self._detected_obstacle_states = None
        self._surrounding_beams = None
        self._detection_points = None
        self.detected_obstacles = None
        self.lanelet_dict = None
        self.all_lanelets_set = None
        self.observation_dict = OrderedDict()

    def build_observation_space(self) -> OrderedDict:
        observation_space_dict = OrderedDict()

        # Lane-based circular/rectangle surrounding observation
        if self.observe_lane_rect_surrounding or self.observe_lane_circ_surrounding:
            observation_space_dict["lane_based_v_rel"] = gymnasium.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
            observation_space_dict["lane_based_p_rel"] = gymnasium.spaces.Box(-self.max_obs_dist, self.max_obs_dist, (6,),
                                                                        dtype=np.float32)
            if self.observe_vehicle_type:
                observation_space_dict["vehicle_type"] = gymnasium.spaces.Box(0, self.num_vehicle_types - 1, (6,),
                                                                        dtype=np.int8)
            if self.observe_vehicle_lights:
                observation_space_dict["vehicle_signals"] = gymnasium.spaces.Box(-1, 1, (6,), dtype=np.int8)
        # Lidar-based elliptical surrounding observation
        elif self.observe_lidar_circle_surrounding:
            num_beams = self.lidar_circle_num_beams
            observation_space_dict["lidar_circle_dist_rate"] = gymnasium.spaces.Box(-np.inf, np.inf, (num_beams,),
                                                                              dtype=np.float32)
            observation_space_dict["lidar_circle_dist"] = gymnasium.spaces.Box(-self.max_obs_dist, self.max_obs_dist,
                                                                         (num_beams,), dtype=np.float32)

            if self.observe_vehicle_type:
                observation_space_dict["vehicle_type"] = gymnasium.spaces.Box(0, self.num_vehicle_types - 1, (num_beams,),
                                                                        dtype=np.int8)
            if self.observe_vehicle_lights:
                observation_space_dict["vehicle_signals"] = gymnasium.spaces.Box(-1, 1, (num_beams,), dtype=np.int8)

            if self.reward_safe_distance_coef != 0:
                observation_space_dict["dist_lead_follow_rel"] = gymnasium.spaces.Box(-self.max_obs_dist, self.max_obs_dist,
                                                                                (2,),
                                                                                dtype=np.float32)
            if self.observe_relative_priority:
                observation_space_dict["rel_prio_lidar"] = gymnasium.spaces.Box(-1, 1, (num_beams,), dtype=np.float32)
        if self.observe_is_collision:
            observation_space_dict["is_collision"] = gymnasium.spaces.Box(0, 1, (1,), dtype=np.float32)

        if self.observe_intersection_velocities:
            observation_space_dict["intersection_velocities"] = gymnasium.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)

        if self.observe_intersection_distances:
            observation_space_dict["intersection_distances"] = gymnasium.spaces.Box(-self.dummy_dist_intersection,
                                                                              self.dummy_dist_intersection, (6,),
                                                                              dtype=np.float32)
        if self.observe_ego_distance_intersection:
            observation_space_dict["ego_distance_intersection"] = gymnasium.spaces.Box(-np.inf, np.inf, (2,),
                                                                                 dtype=np.float32)

        if self.observe_lane_change:
            observation_space_dict["lane_change"] = gymnasium.spaces.Box(0, 1, (1,), dtype=np.float32)
        return observation_space_dict

    def observe(self, scenario: Scenario, ego_vehicle: Vehicle, time_step: int,
                connected_lanelet_dict: Union[None, Dict] = None, ego_lanelet: Union[Lanelet, None] = None,
                collision_checker: Union[None, pycrcc.CollisionChecker] = None,
                local_ccosy: Union[None, CurvilinearCoordinateSystem] = None, 
                conflict_zone: ConflictZone = None,
                ego_lanelet_ids: List[int] = []) -> Union[ndarray, Dict]:

        # update scenario and time_step
        self._scenario = scenario
        self._current_time_step = time_step
        self._ego_state = ego_vehicle.state
        if not self.fast_distance_calculation:
            self._ego_shape = Rectangle(length=ego_vehicle.parameters.l,
                                        width=ego_vehicle.parameters.w,
                                        center=self._ego_state.position,
                                        orientation=self._ego_state.orientation).shapely_object
        self._collision_checker = collision_checker
        self._local_ccosy = local_ccosy
        self.lanelet_dict, self.all_lanelets_set = SurroundingObservation.get_nearby_lanelet_id(connected_lanelet_dict,
                                                                                                ego_lanelet)
        ego_vehicle_lat_position = None

        if self.observe_lane_rect_surrounding or self.observe_lane_circ_surrounding:
            # construct sensing area
            if self.observe_lane_rect_surrounding:
                self._surrounding_area = pycrcc.RectOBB(self.lane_rect_sensor_range_length / 2,
                                                        self.lane_rect_sensor_range_width / 2,
                                                        self._ego_state.orientation if hasattr(
                                                            self._ego_state,
                                                            "orientation") else np.arctan2(
                                                            self._ego_state.velocity_y,
                                                            self._ego_state.velocity),
                                                        self._ego_state.position[0],
                                                        self._ego_state.position[1])
            else:
                self._surrounding_area = pycrcc.Circle(self.lane_circ_sensor_range_radius,
                                                       self._ego_state.position[0],
                                                       self._ego_state.position[1])

            ego_vehicle_lat_position, self._detected_obstacle_states, self.detected_obstacles = \
                self._get_surrounding_obstacles_lane_based(self._surrounding_area)

        elif self.observe_lidar_circle_surrounding:
            self._surrounding_area = pycrcc.Circle(self.lidar_sensor_radius, self._ego_state.position[0],
                                                   self._ego_state.position[1])
            self.detected_obstacles = self._get_surrounding_obstacles_lidar_circle()
            if self.reward_safe_distance_coef != 0:
                self._add_leading_following_distance_lidar_lane()
            if self.observe_relative_priority:
                self._add_relative_priority(self.detected_obstacles, ego_lanelet, ego_vehicle)

        if self.observe_lane_change:
            self._detect_lane_change(ego_lanelet_ids)

        if self.observe_is_collision:
            is_collision = self._check_collision(collision_checker, ego_vehicle)
            self.observation_dict["is_collision"] = np.array([is_collision])

        if self.observe_intersection_velocities or self.observe_intersection_distances:
            if self.observe_lane_circ_surrounding:
                intersection_observation = conflict_zone.generate_intersection_observation(self.conflict_obstacles_information)
                if self.observe_intersection_velocities:
                    self._get_intersection_velocities(intersection_observation)

                if self.observe_intersection_distances:
                    self._get_intersection_distances(intersection_observation)
            else:
                #TODO: Remove this constraint
                print('ERROR:SurroundingObservation: Intersection observations currently only selectable together '
                      'with circle lane based observation')
                raise ValueError

        if self.observe_ego_distance_intersection:
            s_near, s_far = conflict_zone.get_ego_intersection_observation(self._ego_state.position)
            self.observation_dict["ego_distance_intersection"] = np.array([s_near, s_far])

        return self.observation_dict, ego_vehicle_lat_position

    def draw(self, render_configs: Dict, render: MPRenderer, terminated: bool = False):
        # Mark surrounding obstacles (Only if corresponding observations are available)
        # Lane-based surrounding rendering
        # surrounding areas
        if render_configs["render_surrounding_area"] and (
                self.observe_lane_rect_surrounding or self.observe_lane_circ_surrounding):
            self._surrounding_area.draw(render, draw_params={"facecolor": "lightblue",
                                                             "edgecolor": "lightblue", "opacity": 0.5})

        # detected obstacles
        if render_configs["render_surrounding_obstacles_lane_based"] and (
                self.observe_lane_rect_surrounding or self.observe_lane_circ_surrounding):
            colors = ["r", "y", "k", "r", "y", "k"]
            # o_left_follow, o_same_follow, o_right_follow, o_left_lead, o_same_lead, o_right_lead
            for obs, color in zip(self._detected_obstacle_states, colors):
                if obs is not None:
                    render.dynamic_artists.append(
                        LineDataUnits([obs.position[0]], [obs.position[1]], color=color, marker="*",
                                      zorder=ZOrders.OBSTACLES+1, label="surrounding_obstacles_lane_based"))

        # Lidar-based elliptical surrounding rendering
        if self.observe_lidar_circle_surrounding and render_configs["render_lidar_circle_surrounding_beams"]:
            for (beam_start, beam_length, beam_angle) in self._surrounding_beams:
                center = beam_start + 0.5 * beam_length * approx_orientation_vector(beam_angle)
                beam_angle = make_valid_orientation(beam_angle)
                beam_draw_object = Rectangle(length=beam_length, width=0.1, center=center, orientation=beam_angle)
                beam_draw_object.draw(render)

        if self.observe_lidar_circle_surrounding and render_configs["render_lidar_circle_surrounding_obstacles"]:
            for idx, detection_point in enumerate(self._detection_points):
                    render.dynamic_artists.append(
                    LineDataUnits(detection_point[0], detection_point[1], color="b", marker="1",
                                  zorder=ZOrders.INDICATOR_ADD, label="surrounding_obstacles_lidar_based"))

        # TODO: add rendering for intersection observations
        # use conflict_region.draw_conflict_region to highlight region
        # also highlight detected intersection vehicles


    def _get_intersection_velocities(self,intersection_observations: List[State]):
        """
        calculates relative velocity observation for intersection vehicles
        :param intersection_observations: list of intersection obstacle states
        return: relative velocity observations for intersection vehicles
        """
        rel_v = np.zeros((6,))
        for k in range(len(intersection_observations)):
            if k < 6:
                rel_v[k] = intersection_observations[k][1]

        self.observation_dict["intersection_velocities"] = rel_v

    def _get_intersection_distances(self, intersection_observations: List[State]):
        """
        calculates relative distance observation for intersection vehicles
        :param intersection_observations: list of intersection obstacle states
        return: relative distance observations for intersection vehicles
        """
        rel_p = self.dummy_dist_intersection * np.ones((6,))
        for k in range(len(intersection_observations)):
            if k < 6:
                rel_p[k] = min(self.dummy_dist_intersection, intersection_observations[k][0])

        self.observation_dict["intersection_distances"] = rel_p


    def _get_surrounding_obstacles_lane_based(self, surrounding_area: Union[pycrcc.RectOBB, pycrcc.Circle]) \
            -> Tuple[np.array, List[State], List[Obstacle]]:
       
        lanelet_ids, obstacle_states, obstacles = self._get_obstacles_in_surrounding_area(surrounding_area)
        obstacle_lanelet, adj_obstacle_states, adj_obstacles = \
            self._filter_obstacles_in_adj_lanelet(lanelet_ids, obstacle_states, obstacles, self.all_lanelets_set)
        rel_vel, rel_pos, detected_states, detected_obstacles, ego_vehicle_lat_position = \
            self._get_rel_v_p_lane_based(obstacle_lanelet, adj_obstacle_states, self.lanelet_dict, adj_obstacles)

        self.observation_dict["lane_based_v_rel"] = np.array(rel_vel)
        self.observation_dict["lane_based_p_rel"] = np.array(rel_pos)

        if self.observe_vehicle_type:
            self._get_vehicle_types(detected_obstacles)
        if self.observe_vehicle_lights:
            self._get_vehicle_lights(detected_obstacles)


        # add self.conflict_obstacles_information
        detected_states_exclude = [state for state in detected_states if state is not None] # TODO: remove after released cr-io
        self.conflict_obstacles_information = [state for state in obstacle_states if state not in detected_states_exclude]
        # if want to include the 6 vehilces that are already detected: comment the above line and uncomment the following line
        # self.conflict_obstacles_information = [state for state in obstacle_states]

        return ego_vehicle_lat_position, detected_states, detected_obstacles

    def _detect_lane_change(self, ego_lanelet_ids: List[int]) -> None:
        self.observation_dict["lane_change"] = np.array([0.0])
        for lanelet_id in ego_lanelet_ids:
            if lanelet_id not in self.lanelet_dict["ego_all"]:
                self.observation_dict["lane_change"] = np.array([1.0])
                return

    def _add_leading_following_distance_lidar_lane(self) -> None:
        """
        Adds the leading and following obstacle distance observation to dist_lead_follow_rel
        """
        lanelet_ids, obstacle_states, obstacles = self._get_obstacles_in_surrounding_area(self._surrounding_area)
        obstacle_lanelet, adj_obstacle_states, adj_obstacles = \
            self._filter_obstacles_in_adj_lanelet(lanelet_ids, obstacle_states, obstacles, self.all_lanelets_set)
        _, rel_pos, _, _, _ = \
            self._get_rel_v_p_lane_based(obstacle_lanelet, adj_obstacle_states, self.lanelet_dict, adj_obstacles)
        self.observation_dict["dist_lead_follow_rel"] = np.array([rel_pos[4], rel_pos[1]])

    def _add_relative_priority(self, obstacles: List[Obstacle], ego_lanelet: Lanelet,
                               ego_vehicle: Vehicle, turn_threshold=0.02, scan_range=20.):
        """
        Adds relative priority to the observation dict. {-1, 0, 1} for {yield, same, priority} respectively
        :param obstacles: a list of detected obstacles
        """
        # Classify ego trajectory from the reference path (project position onto reference path)
        if self._local_ccosy.cartesian_point_inside_projection_domain(
                self._ego_state.position[0], self._ego_state.position[1]):
            ego_curv_coords = self._local_ccosy.convert_to_curvilinear_coords(self._ego_state.position[0],
                                                                              self._ego_state.position[1])
            ego_proj_pos = self._local_ccosy.convert_to_cartesian_coords(ego_curv_coords[0], 0)
            t_type = self.trajectory_type_from_path(
                self._local_ccosy.reference_path(), turn_threshold, pycrcc.Circle(scan_range, ego_proj_pos[0],
                                                                                  ego_proj_pos[1]))
        # TODO: possible improvement - find the nearest possible point in the domain or use the last state
        #  in the domain as reference
        # If the point is not in the projection domain, classify the entire reference path
        else:
            t_type = self.trajectory_type_from_path(self._local_ccosy.reference_path(), turn_threshold)
        # Calculate ego priority according to its trajectory class and signs
        ego_lanelet_priority = self._detect_lanelet_priority(ego_lanelet, t_type)
        obstacle_rel_lanelet_priorities = []
        # Iterate over the obstacles
        for obs in obstacles:
            if obs is None or obs.obstacle_type == ObstacleRole.STATIC:
                obstacle_rel_lanelet_priorities.append(1.0)
                continue
            obstacle_state = obs.state_at_time(self._current_time_step)
            obs_occupied_lanelet_id = list(obs.prediction.shape_lanelet_assignment[self._current_time_step])
            if len(obs_occupied_lanelet_id) > 1:
                obs_occupied_lanelet_id = Navigator.sorted_lanelet_ids(
                    obs_occupied_lanelet_id, obstacle_state.orientation, obstacle_state.position, self._scenario)[0]
            elif len(obs_occupied_lanelet_id) == 1:
                obs_occupied_lanelet_id = obs_occupied_lanelet_id[0]
            elif len(obs_occupied_lanelet_id) == 0:
                obstacle_rel_lanelet_priorities.append(0.0)
                continue

            obs_lanelet = self._scenario.lanelet_network.find_lanelet_by_id(obs_occupied_lanelet_id)

            obstacle_turning_signal = obs.signal_state_at_time_step(self._current_time_step)

            if obstacle_turning_signal is None:
                obs_t_type = TrajectoryType.STRAIGHT
            elif obstacle_turning_signal.indicator_left:
                obs_t_type = TrajectoryType.LEFT
            elif obstacle_turning_signal.indicator_right:
                obs_t_type = TrajectoryType.RIGHT
            # else:
            #     obs_t_type = TrajectoryType.STRAIGHT

            # Classify the obstacle priority based on its trajectory class and signs
            obstacle_lanelet_priority = self._detect_lanelet_priority(obs_lanelet, obs_t_type)
            intersection_dict = self._scenario.lanelet_network.map_inc_lanelets_to_intersections
            # Check if ego and the obstacle are approaching the same intersection
            # or if the obstacle is already in an intersection
            if ego_lanelet.lanelet_id in intersection_dict \
                    and ((obs_lanelet.lanelet_id in intersection_dict and
                          intersection_dict[ego_lanelet.lanelet_id] == intersection_dict[obs_lanelet.lanelet_id])
                         or obs_lanelet.lanelet_type == LaneletType.INTERSECTION):
                # If an obstacle is in the middle of an intersection -> yield
                if obs_lanelet.lanelet_type == LaneletType.INTERSECTION:
                    obstacle_rel_lanelet_priorities.append(-1.0)
                # Obstacle in the same lanelet as ego -> same priority
                # TODO: include adjacent lanelets -> obstacles can be ignored as well
                elif obs_lanelet.lanelet_id == ego_lanelet.lanelet_id:
                    obstacle_rel_lanelet_priorities.append(0.0)
                # The signs match -> check right before left
                elif ego_lanelet_priority in [4, 5, 6] and obstacle_lanelet_priority in [4, 5, 6] or \
                        self._matching_signs(ego_lanelet, obs_lanelet):
                    # Right before left rule applies, check by orientation
                    obs_state = obs.state_at_time(self._current_time_step)
                    ego_state = ego_vehicle.state
                    ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else \
                        np.arctan2(ego_state.velocity_y, ego_state.velocity)
                    rel_orientation = make_valid_orientation(obs_state.orientation - ego_state_orientation)
                    if np.isclose(rel_orientation, np.pi / 2, atol=np.pi * 1 / 6):
                        obstacle_rel_lanelet_priorities.append(-1.0)
                        continue
                    elif np.isclose(rel_orientation, 1.5 * np.pi, atol=np.pi * 1 / 6):
                        obstacle_rel_lanelet_priorities.append(1.0)
                        continue
                    # obstacle_rel_lanelet_priorities.append(-1.0)
                # Check sign + driving direction rules
                if obstacle_lanelet_priority < ego_lanelet_priority:
                    obstacle_rel_lanelet_priorities.append(1.0)
                elif obstacle_lanelet_priority > ego_lanelet_priority:
                    obstacle_rel_lanelet_priorities.append(-1.0)
                else:
                    obstacle_rel_lanelet_priorities.append(0.0)
            else:
                obstacle_rel_lanelet_priorities.append(1.0)
        self.observation_dict["rel_prio_lidar"] = np.array(obstacle_rel_lanelet_priorities)

    @staticmethod
    def trajectory_type_from_path(ref_path: ndarray, turn_threshold=0.02, traj_area=None) -> TrajectoryType:
        """
        Classifies the trajectory of a path (2d polyline) at the turn threshold. Filters points that aren't in
        the surrounding area
        :param ref_path: 2d polyline path that gets classified
        :param turn_threshold: the minimum curvature of the path to be classified as a turn
        :param traj_area: shapely object of the area in which the trajectory's analyzed
        """
        ref_path_before = ref_path
        if traj_area:
            ref_path_points = []
            for point in ref_path:
                obstacle_point = pycrcc.Point(point[0], point[1])
                if traj_area.collide(obstacle_point):
                    ref_path_points.append(point)
            ref_path = ref_path_points

        state_list = []
        dummy_velocity = 1.0
        time_step = 0
        for point in ref_path:
            state_list.append(CustomState(position=point, velocity=dummy_velocity, time_step=time_step))
            time_step += 1
        assert len(state_list) != 0, "ref_path is " + str(ref_path) + " vs before " + str(
            ref_path_before) + " shape: (r,x,y)" + str(traj_area.r()) + " " + str(traj_area.x()) + " " + str(
            traj_area.y())
        traj = Trajectory(0, state_list)
        t_type = classify_trajectory(traj, min_velocity=dummy_velocity - 1.0, turn_threshold=turn_threshold)
        return t_type

    def _matching_signs(self, ego_lanelet: Lanelet, obs_lanelet: Lanelet):
        sign_ids_obs = list(self._scenario.lanelet_network.find_traffic_sign_by_id(s).
                            traffic_sign_elements[0].traffic_sign_element_id for s in obs_lanelet.traffic_signs)
        sign_ids_ego = list(self._scenario.lanelet_network.find_traffic_sign_by_id(s).
                            traffic_sign_elements[0].traffic_sign_element_id for s in ego_lanelet.traffic_signs)
        return any(s_id == s_id_ego and s_id in [TrafficSignIDGermany.YIELD, TrafficSignIDGermany.STOP,
                                                 TrafficSignIDGermany.RIGHT_OF_WAY, TrafficSignIDGermany.PRIORITY]
                   for s_id in sign_ids_obs for s_id_ego in sign_ids_ego)

    def _detect_lanelet_priority(self, lanelet: Lanelet, traj_type: TrajectoryType) -> int:
        """
        Returns a priority number between 1 and 9. The larger the number, the earlier the vehicle is allowed to drive
        Only includes lanelet priority and trajectory (no right before left rule)
        """
        sign_ids = list(self._scenario.lanelet_network.find_traffic_sign_by_id(s).
                        traffic_sign_elements[0].traffic_sign_element_id for s in lanelet.traffic_signs)

        if any(s_id in [TrafficSignIDGermany.YIELD, TrafficSignIDGermany.STOP] for s_id in sign_ids):
            if traj_type == TrajectoryType.LEFT:
                return 1
            elif traj_type == TrajectoryType.RIGHT:
                return 2
            else:
                return 3
        elif any(s_id in [TrafficSignIDGermany.RIGHT_OF_WAY, TrafficSignIDGermany.PRIORITY] for s_id in sign_ids):
            if traj_type == TrajectoryType.LEFT:
                return 7
            elif traj_type == TrajectoryType.RIGHT:
                return 8
            else:
                return 9
        else:
            if traj_type == TrajectoryType.LEFT:
                return 4
            elif traj_type == TrajectoryType.RIGHT:
                return 5
            else:
                return 6

    def _get_surrounding_obstacles_lidar_circle(self) -> List[Obstacle]:

        obstacle_shapes, detected_obstacles = self._get_obstacle_shapes_in_surrounding_area(self._surrounding_area)

        # Create beam shapes (shapely line strings) around the ego vehicle, forming an ellipse sensing area as a whole
        surrounding_beams_ego_vehicle = []
        beam_start = self._ego_state.position
        for i in range(self.lidar_circle_num_beams):
            theta = i * (2 * np.pi / self.lidar_circle_num_beams)
            x_delta = self.lidar_sensor_radius * np.cos(theta)
            y_delta = self.lidar_sensor_radius * np.sin(theta)
            beam_length = np.sqrt(x_delta ** 2 + y_delta ** 2)
            beam_angle = self._ego_state.orientation + theta
            surrounding_beams_ego_vehicle.append((beam_start, beam_length, beam_angle))

        obstacle_distances, observed_obstacles \
            = self._get_obstacles_with_surrounding_beams(obstacle_shapes, detected_obstacles,
                                                         surrounding_beams_ego_vehicle)

        distances, distance_rates, detection_points = self._get_distances_lidar_based(surrounding_beams_ego_vehicle,
                                                                                      obstacle_distances)

        self.observation_dict["lidar_circle_dist_rate"] = np.array(distance_rates)
        self.observation_dict["lidar_circle_dist"] = np.array(distances)

        if self.observe_vehicle_type:
            self._get_vehicle_types(observed_obstacles)
        if self.observe_vehicle_lights:
            self._get_vehicle_lights(observed_obstacles)
        self._surrounding_beams = surrounding_beams_ego_vehicle
        self._detection_points = detection_points

        return observed_obstacles

    @staticmethod
    def _check_collision(collision_checker: pycrcc.CollisionChecker, ego_vehicle: Vehicle) -> bool:
        collision_ego_vehicle = ego_vehicle.collision_object
        return collision_checker.collide(collision_ego_vehicle)

    def _get_obstacle_shapes_in_surrounding_area(self, surrounding_area: pycrcc.Shape) \
            -> Tuple[List[Polygon], List[Obstacle]]:
        """
        Get the occupancy shape and states and lanelet ids of all obstacles
        within the range of surrounding area of ego vehicle.
        :param surrounding_area: Shapes of pycrcc classes
        :return: List of obstacle shapely shapes
        """
        obstacle_shapes = []
        detected_obstacles = []
        dyn_obstacles, static_obstacles = self._scenario.dynamic_obstacles, self._scenario.static_obstacles

        for o in dyn_obstacles:
            if o.initial_state.time_step <= self._current_time_step <= o.prediction.trajectory.final_state.time_step:
                obstacle_state = o.state_at_time(self._current_time_step)
                obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
                if surrounding_area.collide(obstacle_point):
                    obstacle_shapes.append(o.occupancy_at_time(self._current_time_step).shape.shapely_object)
                    detected_obstacles.append(o)

        for o in static_obstacles:
            obstacle_state = o.initial_state
            obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
            if surrounding_area.collide(obstacle_point):
                obstacle_shapes.append(o.occupancy_at_time(self._current_time_step).shape.shapely_object)
                detected_obstacles.append(o)

        return obstacle_shapes, detected_obstacles

    def _get_obstacles_with_surrounding_beams(self, obstacle_shapes: List[Polygon], obstacles: List[Obstacle],
                                              surrounding_beams: List[Tuple[float, float, float]]) \
            -> Tuple[np.ndarray, List[Optional[Obstacle]]]:
        """
        Get the distance to the nearest obstacles colliding with LIDAR beams

        :param obstacle_shapes: Obstacle shapes that detected with given sensing area
        :param obstacles: The obstacles belonging to the obstacle_shapes
        :param surrounding_beams: List of beams as start point, length and angle
        :return: List of obstacle states
        """
        obstacle_distances = np.zeros(len(surrounding_beams))
        observed_obstacles = [None] * len(surrounding_beams)
        # For each beam, record all collisions with obstacles first, and report the one being closest to the ego vehicle
        ego_vehicle_center_shape = Point(self._ego_state.position)
        for (i, (beam_start, beam_length, beam_angle)) in enumerate(surrounding_beams):
            beam_vec = approx_orientation_vector(beam_angle) * beam_length
            # asLineString recycles C-array as explained
            # beam = asLineString(np.array([beam_start, beam_start + beam_vec]))
            beam = LineString([beam_start, beam_start + beam_vec])
            obstacle_candidate_dist = self.max_obs_dist
            obstacle_candidate = None

            for j, obstacle_shape in enumerate(obstacle_shapes):
                # TODO also support Shapegroups without shapely_object attribute
                if beam.intersects(obstacle_shape):
                    dist = ego_vehicle_center_shape.distance(beam.intersection(obstacle_shape))
                    if dist < obstacle_candidate_dist:
                        obstacle_candidate_dist = dist
                        obstacle_candidate = obstacles[j]

            obstacle_distances[i] = obstacle_candidate_dist
            observed_obstacles[i] = obstacle_candidate

        return obstacle_distances, observed_obstacles

    def _get_distances_lidar_based(self, beams: List[Tuple[np.ndarray, float, float]],
                                   obstacle_distances: Union[List[float], np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        calculate the points in the obstacle where the lidar beam hit
        calculate the length/distance of the ego_state to that point
        and calculate the change of the distances to the obstacles over the last time_step

        :param beams: List of beams as start coordinates, length and angle
        :param obstacle_distances: distance of the beam to the detected point
            if it is equal to self.lidar_sensor_radius no object was detected
        :return:
            dists: same as input obstacle_distances
            dist_rates: change of the distances to the detected obstacle normalized by the time step
            detection_points: point where the beam hit the obstacle

        Examples::
            beams = [(np.array([0,0]),10,0)]
            obstacle_distances = [3]
            dists, dist_rates, detection_points = surrounding_observation._get_distances_lidar_based(beams,
            obstacle_distances)
        """
        # Relative positions
        dists = np.array(obstacle_distances)
        prev_distances = self.observation_dict.get("lidar_circle_dist", np.full(len(beams), self.max_obs_dist))

        # Change rates
        if self._current_time_step == 0:
            dist_rates = np.full(len(beams), 0.0)
        else:
            dist_rates = (prev_distances - dists) / self._scenario.dt

        # detection points
        detection_points = [beam_start + approx_orientation_vector(beam_angle) * dists[i] for
                            i, ((beam_start, _, beam_angle), closest_collision) in
                            enumerate(zip(beams, obstacle_distances))]

        return dists, dist_rates, detection_points

    def _get_obstacles_in_surrounding_area(self, surrounding_area: pycrcc.Shape) \
            -> Tuple[List[int], List[State], List[Obstacle]]:
        """
        Get the states and lanelet ids of all obstacles within the range of surrounding area of ego vehicle.

        :return: List of lanelet ids of obstacles, list of states obstacles
        """
        lanelet_ids, obstacle_states, obstacles = [], [], []
        dyn_obstacles, static_obstacles = self._scenario.dynamic_obstacles, self._scenario.static_obstacles

        # iterate over all dynamic obstacles
        for o in dyn_obstacles:
            if o.obstacle_id == -1:
                continue

            # TODO: initial lanelet_assignment missed?
            if o.prediction is not None:
                center_lanelet_ids = list(o.prediction.center_lanelet_assignment.values())
                if o.initial_state.time_step <= self._current_time_step <= o.prediction.trajectory.final_state.time_step:
                    obstacle_state = o.state_at_time(self._current_time_step)
                    obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
                    if surrounding_area.collide(obstacle_point):
                        # o_center_lanelet_ids = list(o.prediction.center_lanelet_assignment[self._current_time_step])
                        o_shape_lanelet_ids = list(o.prediction.shape_lanelet_assignment[self._current_time_step])
                        o_lanelet_id = None # self._get_occupied_lanelet_id(self._scenario, o_center_lanelet_ids, obstacle_state)
                        if o_lanelet_id is None:
                            # use shape lanelet assignment instead
                            o_lanelet_id = self._get_occupied_lanelet_id(self._scenario, o_shape_lanelet_ids,
                                                                         obstacle_state)
                            if o_lanelet_id is None:
                                # neither center or shape locate inside a lanelet
                                # TODO: skip or take last available time step ??
                                non_empty_id_sets = [id_set for id_set in center_lanelet_ids if id_set]
                                if len(non_empty_id_sets) == 0:
                                    continue
                                o_lanelet_id = self._get_occupied_lanelet_id(self._scenario, list(non_empty_id_sets[-1]),
                                                                             obstacle_state)
                        lanelet_ids.append(o_lanelet_id)
                        obstacle_states.append(obstacle_state)
                        obstacles.append(o)
            else: # obstacle only has initial state (interactive scenarios)
                if o.initial_state.time_step == self._current_time_step:
                    obstacle_state = o.initial_state
                    obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
                    if surrounding_area.collide(obstacle_point):
                        o_center_lanelet_ids = list(o.initial_center_lanelet_ids)
                        o_shape_lanelet_ids = list(o.initial_shape_lanelet_ids)
                        # sort lanelet ids by orientation and relevance
                        o_lanelet_id = self._get_occupied_lanelet_id(self._scenario, o_center_lanelet_ids, obstacle_state)
                        if o_lanelet_id is None:
                            # use shape lanelet assignment instead
                            o_lanelet_id = self._get_occupied_lanelet_id(self._scenario, o_shape_lanelet_ids,
                                                                         obstacle_state)
                            if o_lanelet_id is None:
                                # neither center or shape locate inside a lanelet
                                continue
                        lanelet_ids.append(o_lanelet_id)
                        obstacle_states.append(obstacle_state)
                        obstacles.append(o)

        # iterate over all static obstacles
        for o in static_obstacles:
            obstacle_state = o.initial_state
            obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
            if surrounding_area.collide(obstacle_point):
                obstacle_lanelet_ids = list(o.initial_center_lanelet_ids)
                lanelet_id = self._get_occupied_lanelet_id(self._scenario, obstacle_lanelet_ids, obstacle_state)
                lanelet_ids.append(lanelet_id)
                obstacle_states.append(obstacle_state)
                obstacles.append(o)

        return lanelet_ids, obstacle_states, obstacles

    def _get_vehicle_types(self, obstacles: List[Optional[Obstacle]]) -> None:
        """
        Sets the obstacle types in observation_dict for all observed obstacles
        {0,1,2,3,4,5,6} = {Other, Car, Bicycle, Pedestrian, Truck, Bus, Static}

        Note: If the mapping of types to numbers is changed, self.num_vehicle_types should be adjusted accordingly

        :param obstacles: The observed obstacles
        """
        types = [0] * len(obstacles)

        for i, obstacle in enumerate(obstacles):
            if obstacle is None:
                continue

            if obstacle.obstacle_role == ObstacleRole.STATIC:
                types[i] = 6
            elif obstacle.obstacle_type == ObstacleType.CAR:
                types[i] = 1
            elif obstacle.obstacle_type == ObstacleType.BICYCLE:
                types[i] = 2
            elif obstacle.obstacle_type == ObstacleType.PEDESTRIAN:
                types[i] = 3
            elif obstacle.obstacle_type == ObstacleType.TRUCK:
                types[i] = 4
            elif obstacle.obstacle_type == ObstacleType.BUS:
                types[i] = 5

        self.observation_dict["vehicle_type"] = np.array(types)

    def _get_vehicle_lights(self, obstacles: List[Optional[Obstacle]]) -> None:
        """
Im        Sets the turning-lights in observation_dict for all observed obstacles
        {-1, 0, 1} = {Left signal, Off, Right signal}

        :param obstacles: The observed obstacles
        """
        signals = [0] * len(obstacles)

        for i, obstacle in enumerate(obstacles):
            if obstacle is None:
                continue

            signal_state: SignalState = obstacle.signal_state_at_time_step(self._current_time_step)
            if signal_state is None:
                continue
            if signal_state.indicator_right:
                signals[i] = 1
            elif signal_state.indicator_left:
                signals[i] = -1

        self.observation_dict["vehicle_signals"] = np.array(signals)

    @staticmethod
    def _get_occupied_lanelet_id(scenario: Scenario, obstacle_lanelet_ids: List[int], obstacle_state: State) \
            -> Union[None, int]:
        """
        gets most relevant lanelet id from obstacle_lanelet_ids for an obstacle that occupies multiple lanelets

        :param scenario: current scenario
        :param obstacle_lanelet_ids: lanelet ids of lanelets occupied by the obstacle
        :param obstacle_state: current state of the obstacle
        """
        if len(obstacle_lanelet_ids) > 1:
            # select the most relevant lanelet
            return Navigator.sorted_lanelet_ids(
                obstacle_lanelet_ids, obstacle_state.orientation, obstacle_state.position, scenario)[0]
        elif len(obstacle_lanelet_ids) == 1:
            return obstacle_lanelet_ids[0]
        else:
            return None

    @staticmethod
    def _filter_obstacles_in_adj_lanelet(lanelet_ids: List[int], states: List[State], obstacles: List[Obstacle],
                                         all_lanelets_set: Set[int]) -> Tuple[List[int], List[State], List[Obstacle]]:
        """
        filters out obstacles states and their corresponding lanelet id
        where the lanelet id is not in the all_lanelets_set

        :param lanelet_ids: List of lanelet ids of obstacles
        :param states: List of states of obstacles
        :param obstacles: List of obstacles
        :param all_lanelets_set: The set of all lanelet ids in the scenario
        :return: The list of lanelets of obstacles, the list of states
        """
        adj_obstacle_states, obstacle_lanelet, adj_obstacles = [], [], []
        for lanelet_id, state, obstacle in zip(lanelet_ids, states, obstacles):
            if lanelet_id in all_lanelets_set:  # Check if the obstacle is in adj lanelets
                obstacle_lanelet.append(lanelet_id)
                adj_obstacle_states.append(state)
                adj_obstacles.append(obstacle)

        return obstacle_lanelet, adj_obstacle_states, adj_obstacles

    @staticmethod
    def get_rel_v_p_follow_leading(distance_sign: int, distance_abs: float, p_rel_follow: float, p_rel_lead: float,
                                   v_rel_follow: float, v_rel_lead: float, obs_state: State, obstacle: Obstacle,
                                   ego_state: State, o_follow: State, o_lead: State, obstacle_follow: Obstacle,
                                   obstacle_lead: Obstacle) -> \
            Tuple[float, float, State, Obstacle, float, float, State, Obstacle]:
        """
        #TODO maybe change signature to only have a single variable for each leading and following
            e.g. instead of o_follow, o_lead just o
        calculates the relative velocity of leading and following obstacles to the ego vehicle

        :param distance_sign: 1 -> follow, !=1 -> lead
        :param distance_abs: absolut distance of ego vehicle to obstacle
        :param p_rel_follow: max distance of a obstacle that is following the ego vehicle
        :param p_rel_lead: max distance of a obstacle that is leading the ego vehicle
        :param v_rel_follow: relative velocity of the following obstacle
        :param v_rel_lead: relative velocity of the leading obstacle
        :param obs_state: state of the obstacle
        :param ego_state: state of the ego vehicle
        :param o_follow: if the obs_state is following it is stored here
        :param o_lead: if the obs_state is leading it is stored here

        :return
            v_rel_follow: relative velocity to following obstacle
            p_rel_follow: relative position to following obstacle
            o_follow: state of following obstacle
            v_rel_lead: relative velocity to leading obstacle
            p_rel_lead: relative position to leading obstacle
            o_lead: state of leading obstacle
        """
        if isinstance(obstacle, StaticObstacle):
            obs_state.velocity = 0.
        if distance_sign == 1 and distance_abs < p_rel_follow:
            # following vehicle, distance is smaller
            ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else np.arctan2(
                ego_state.velocity_y, ego_state.velocity)
            delta_orientation = obs_state.orientation - ego_state_orientation
            v_rel_follow = ego_state.velocity - obs_state.velocity * np.cos(delta_orientation)
            p_rel_follow = distance_abs
            o_follow = obs_state
            obstacle_follow = obstacle
        elif distance_sign != 1 and distance_abs < p_rel_lead:
            # leading vehicle, distance is smaller
            ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else np.arctan2(
                ego_state.velocity_y, ego_state.velocity)
            delta_orientation = obs_state.orientation - ego_state_orientation
            v_rel_lead = obs_state.velocity * np.cos(delta_orientation) - ego_state.velocity
            p_rel_lead = distance_abs
            o_lead = obs_state
            obstacle_lead = obstacle

        return v_rel_follow, p_rel_follow, o_follow, obstacle_follow, v_rel_lead, p_rel_lead, o_lead, obstacle_lead

    def _get_rel_v_p_lane_based(
            self, obstacles_lanelet_ids: List[int], obstacle_states: List[State], lanelet_dict: Dict[str, List[int]],
            adj_obstacles: List[Obstacle]) -> Tuple[List[float], List[float], List[State], List[Obstacle], np.array]:

        """
        Get the relative velocity and position of obstacles in adj left, adj right and ego lanelet.
        In each lanelet, compute only the nearest leading and following obstacles.

        :param obstacles_lanelet_ids: The list of lanelets of obstacles
        :param obstacle_states: The list of states of obstacles
        :param lanelet_dict: The lanelet dictionary
            stores the list of lanelet ids by given keywords as (ego_all, ego_right....)
        :param adj_obstacles: lane-based adjacent obstacles
        :return: Relative velocities, relative positions, and detected obstacle states
        """
        # Initialize dummy values, in case no obstacles are present
        v_rel_left_follow, v_rel_same_follow, v_rel_right_follow, v_rel_left_lead, v_rel_same_lead, \
        v_rel_right_lead = [0.0] * 6

        p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead, \
        p_rel_right_lead = [self.max_obs_dist] * 6

        try:
            ego_vehicle_long_position, ego_vehicle_lat_position = self._local_ccosy.convert_to_curvilinear_coords(
                self._ego_state.position[0], self._ego_state.position[1])

            o_left_follow, o_left_lead, o_right_follow, o_right_lead, o_same_follow, o_same_lead = [None] * 6
            obstacle_left_follow, obstacle_left_lead, obstacle_right_follow, obstacle_right_lead, \
            obstacle_same_follow, obstacle_same_lead = [None] * 6

            for o_state, o_lanelet_id, obstacle in zip(obstacle_states, obstacles_lanelet_ids, adj_obstacles):

                distance_abs, distance_sign = self._get_ego_obstacle_distance(obstacle, o_state,
                                                                              (ego_vehicle_long_position,
                                                                               ego_vehicle_lat_position))

                if o_lanelet_id in lanelet_dict["ego_all"]:  # ego lanelet
                    v_rel_same_follow, p_rel_same_follow, o_same_follow, obstacle_same_follow, \
                    v_rel_same_lead, p_rel_same_lead, o_same_lead, obstacle_same_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_same_follow, p_rel_same_lead, v_rel_same_follow,
                            v_rel_same_lead, o_state, obstacle, self._ego_state, o_same_follow, o_same_lead,
                            obstacle_same_follow, obstacle_same_lead)

                if o_lanelet_id in lanelet_dict["right_all"]:  # right lanelet
                    v_rel_right_follow, p_rel_right_follow, o_right_follow, obstacle_right_follow, \
                    v_rel_right_lead, p_rel_right_lead, o_right_lead, obstacle_right_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_right_follow, p_rel_right_lead, v_rel_right_follow,
                            v_rel_right_lead, o_state, obstacle, self._ego_state, o_right_follow, o_right_lead,
                            obstacle_right_follow, obstacle_right_lead)

                if o_lanelet_id in lanelet_dict["left_all"]:  # left lanelet
                    v_rel_left_follow, p_rel_left_follow, o_left_follow, obstacle_left_follow, \
                    v_rel_left_lead, p_rel_left_lead, o_left_lead, obstacle_left_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_left_follow, p_rel_left_lead, v_rel_left_follow,
                            v_rel_left_lead, o_state, obstacle, self._ego_state, o_left_follow, o_left_lead,
                            obstacle_left_follow, obstacle_left_lead)

            detected_states = [o_left_follow, o_same_follow, o_right_follow, o_left_lead, o_same_lead, o_right_lead]
            detected_obstacles = [obstacle_left_follow, obstacle_same_follow, obstacle_right_follow,
                                  obstacle_left_lead, obstacle_same_lead, obstacle_right_lead]

        except ValueError:
            detected_states = [None] * 6
            detected_obstacles = [None] * 6
            ego_vehicle_lat_position = None

        v_rel = [v_rel_left_follow, v_rel_same_follow, v_rel_right_follow, v_rel_left_lead, v_rel_same_lead,
                 v_rel_right_lead]
        p_rel = [p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead,
                 p_rel_right_lead]

        return v_rel, p_rel, detected_states, detected_obstacles, ego_vehicle_lat_position

    def _get_ego_obstacle_distance(self, obstacle: Obstacle, obstacle_state: State, ego_curvi: Tuple[float, float]) \
            -> Tuple[float, int]:
        """
        Get the distance between the ego_vehicle and an obstacle

        :param obstacle: The obstacle in question
        :param ego_curvi: The position of the ego_vehicle in the curvi-system
        :returns: The absolute distance between the shapes
                / The absolute distance between the curvi_long positions depending on the configuration
                , The distance sign
        """

        ego_curvi_long_position, _ = ego_curvi

        try:
            o_curvi_long_position, _ = self._local_ccosy.convert_to_curvilinear_coords(obstacle_state.position[0],
                                                                                       obstacle_state.position[1])
        except ValueError:
            # the position is out of project area of curvilinear coordinate system
            o_curvi_long_position = ego_curvi_long_position + self.max_obs_dist

        distance_sign = np.sign(ego_curvi_long_position - o_curvi_long_position)

        if self.fast_distance_calculation:
            dist_abs = np.abs(ego_curvi_long_position - o_curvi_long_position)

        else:
            o_shape = obstacle.occupancy_at_time(self._current_time_step).shape.shapely_object
            dist_abs = self._ego_shape.distance(o_shape)

        return dist_abs, distance_sign

    @staticmethod
    def get_nearby_lanelet_id(connected_lanelet_dict: dict, ego_vehicle_lanelet: Lanelet) -> Tuple[dict, set]:
        """
        Get ids of nearby lanelets, e.g. lanelets that are successors, predecessors, left, or right of the
        `ego_vehicle_lanelet`
        additionally, all the connected lanelets of the nearby lanes are added. Connected lanelets are defined
        in the `connected_lanelet_dict`

        :param connected_lanelet_dict: A dict with its keys as lanelet id and values as connected lanelet ids
        :param ego_vehicle_lanelet: The list lanelets of the ego vehicle
        :return: A dict of nearby lanelets ids and the set of all nearby lanelets ids.
        """
        keys = {"ego", "left", "right", "ego_other", "left_other", "right_other", "ego_all", "left_all", "right_all", }
        lanelet_dict = {key: set() for key in keys}
        ego_vehicle_lanelet_id = ego_vehicle_lanelet.lanelet_id
        lanelet_dict["ego"].add(ego_vehicle_lanelet_id)  # Add ego lanelet

        for predecessor_lanelet_id in ego_vehicle_lanelet.predecessor:
            lanelet_dict["ego_other"].update(connected_lanelet_dict[predecessor_lanelet_id])
        for successor_lanelet_id in ego_vehicle_lanelet.successor:
            lanelet_dict["ego_other"].update(connected_lanelet_dict[successor_lanelet_id])

        if ego_vehicle_lanelet.adj_right_same_direction:
            # Get adj right lanelet with same direction
            lanelet_dict["right"].add(ego_vehicle_lanelet.adj_right)
        if ego_vehicle_lanelet.adj_left_same_direction:
            # Get adj left lanelet with same direction
            lanelet_dict["left"].add(ego_vehicle_lanelet.adj_left)

        for ego_lanelet_id in lanelet_dict["ego"]:
            lanelet_dict["ego_other"].update(connected_lanelet_dict[ego_lanelet_id])
        for left_lanelet_id in lanelet_dict["left"]:
            lanelet_dict["left_other"].update(connected_lanelet_dict[left_lanelet_id])
        for r in lanelet_dict["right"]:
            lanelet_dict["right_other"].update(connected_lanelet_dict[r])

        lanelet_dict["ego_all"] = set().union(set(lanelet_dict["ego"]), set(lanelet_dict["ego_other"]))

        lanelet_dict["left_all"] = set().union(set(lanelet_dict["left"]), set(lanelet_dict["left_other"]))
        lanelet_dict["right_all"] = set().union(set(lanelet_dict["right"]), set(lanelet_dict["right_other"]))

        all_lanelets_set = set().union(lanelet_dict["ego_all"], lanelet_dict["left_all"], lanelet_dict["right_all"])

        return lanelet_dict, all_lanelets_set


if __name__ == "__main__":
    import yaml
    from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

    config_file = PATH_PARAMS["configs"]["commonroad-v1"]
    with open(config_file, "r") as config_file:
        config = yaml.safe_load(config_file)
    configs = config["env_configs"]
    surrounding_observation = SurroundingObservation(configs)
    print(surrounding_observation)
