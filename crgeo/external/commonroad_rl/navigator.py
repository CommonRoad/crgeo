"""
Navigation connecting the RoutePlanner with Observations
"""
import logging
import math
import warnings
from enum import Enum
from typing import List, Set, Tuple, Union

import commonroad.geometry.shape as cr_shape
import commonroad_dc.pycrccosy as pycrccosy
import numpy as np
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.geometry.util import compute_polyline_length, resample_polyline
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.common.geometry.helpers import TWO_PI
from commonroad_route_planner.route_planner import RoutePlanner, RouteType

logger = logging.getLogger(__name__)


class Navigator(AutoReprMixin):
    # Adapted version of Navigator class included in CommonRoad-RL (https://commonroad.in.tum.de/commonroad-rl)

    class CosyVehicleObservation(Enum):
        """Enum for the observations CoSy"""
        # TODO: add this as a property for the class, not for every function call.

        AUTOMATIC = "automatic"
        LOCALCARTESIAN = "local_cartesian"
        VEHICLEFRAME = "vehicle_frame_cosy"

        @classmethod
        def values(cls):
            return [item.value for item in cls]

    def __init__(
        self, 
        scenario: Scenario,
        planning_problem: PlanningProblem
    ):
        self.scenario = scenario
        self.lanelet_network = self.scenario.lanelet_network
        self.planning_problem = planning_problem

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            route_planner = RoutePlanner(
                scenario=scenario,
                planning_problem=planning_problem,
                backend=RoutePlanner.Backend.NETWORKX,
                log_to_console=False
            )
            route_planner.logger.setLevel(logging.ERROR)
            route_candidates = route_planner.plan_routes()
            route = route_candidates.retrieve_best_route_by_orientation()

        if route is None:
            warnings.warn("Navigator unable to find route")
            return

        self.route = route

        self.sectionized_environment = self.route.retrieve_route_sections()
        self.sectionized_environment_set = set(
            [item for sublist in self.sectionized_environment for item in sublist]
        )
        self.merged_route_lanelets = None
        self.ccosy_list = self._get_route_cosy()

        self.num_of_lane_changes = len(self.ccosy_list)
        self.merged_section_length_list = np.array(
            [curvi_cosy.length() for curvi_cosy in self.ccosy_list]
        )
        # ================================= #
        #         Goal         #
        # ================================= #

        self._initialize_goal()

        # version 06.2021 variables
        # ================================= #
        #    Curvy Lane Dict #
        # ================================= #
        self.lane_ccosy_kd_ref_dict = {}
        self._initialize_lane_ccosy_kd_ref_dict()
        self._initialized_goal_ref_path()

    def _initialize_goal(self):
        # version 2020
        self.goal_curvi_face_coords = None

        if self.route.type == RouteType.REGULAR:
            goal_face_coords = self._get_goal_face_points(
                self._get_goal_polygon(self.planning_problem.goal)
            )
            self.goal_curvi_face_coords = np.array(
                [
                    (self._get_safe_curvilinear_coords(self.ccosy_list[-1], g))[0]
                    for g in goal_face_coords
                ]
            )

            self.goal_min_curvi_coords = np.min(self.goal_curvi_face_coords, axis=0)
            self.goal_max_curvi_coord = np.max(self.goal_curvi_face_coords, axis=0)

    def _initialized_goal_ref_path(self):
        """
        computes the longituginal coordinates of the goal on the route.reference path
        """

        # version 06.2021
        # default for Survival Scenarios: Goal is everywhere reached.
        self.goal_min_curvi_coords_ref = -np.inf
        self.goal_max_curvi_coord_ref = np.inf

        if self.route.type == RouteType.REGULAR:
            # for Regular Scenarios: Goal is everywhere reached.
            goal_face_coords = self._get_goal_face_points(
                self._get_goal_polygon(self.planning_problem.goal)
            )
            self.goal_curvi_face_coords_ref = np.array(
                [
                    (
                        self._get_safe_distance_to_curvilinear2(
                            id_curvilinear="reference_path", position_ego=pos_g
                        )
                    )[0]
                    for pos_g in goal_face_coords
                ]
            )
            self.goal_min_curvi_coords_ref = np.amin(self.goal_curvi_face_coords_ref)
            self.goal_max_curvi_coord_ref = np.amax(self.goal_curvi_face_coords_ref)

    def _initialize_lane_ccosy_kd_ref_dict(self):
        """
        initializes the lane_ccosy_kd_ref_dict with Curvilinearsystems and kdtree
        """
        # version 06.2021
        id_polyline = [("reference_path", self.route.reference_path)]
        for id_lanelet in self.route.list_ids_lanelets:
            id_polyline.append(
                (
                    id_lanelet,
                    self.route.scenario.lanelet_network.find_lanelet_by_id(
                        id_lanelet
                    ).center_vertices,
                )
            )

        for id_curvilinear, polyline in id_polyline:

            ccosy, resampled_polyline = self.create_coordinate_system_from_polyline(
                polyline, return_resampled_polyline=True
            )

            self.lane_ccosy_kd_ref_dict[id_curvilinear] = (
                ccosy,
                KDTree(resampled_polyline),
                resampled_polyline,
            )

    def _get_route_cosy(
        self,
    ) -> Union[pycrccosy.CurvilinearCoordinateSystem, List[Lanelet]]:
        """
        merges the lanelets of reference path from start to goal to successing lanelets
        """
        # version 2020
        self.merged_route_lanelets = []

        # Append predecessor of the initial to ensure that the goal state is not out of the projection domain
        # initial_lanelet = self.lanelet_network.find_lanelet_by_id(self.route.route[0])
        # predecessors_lanelet = initial_lanelet.predecessor
        # if predecessors_lanelet is not None and len(predecessors_lanelet) != 0:
        #     predecessor_lanelet = self.lanelet_network.find_lanelet_by_id(predecessors_lanelet[0])
        #     current_merged_lanelet = predecessor_lanelet
        # else:
        #     current_merged_lanelet = None
        current_merged_lanelet = None

        for current_lanelet_id, next_lanelet_id in zip(
            self.route.list_ids_lanelets[:-1], self.route.list_ids_lanelets[1:]
        ):
            lanelet = self.lanelet_network.find_lanelet_by_id(current_lanelet_id)
            # If the lanelet is the end of a section, then change section
            if next_lanelet_id not in lanelet.successor:
                if current_merged_lanelet is not None:
                    self.merged_route_lanelets.append(current_merged_lanelet)
                    current_merged_lanelet = None
            else:
                if current_merged_lanelet is None:
                    current_merged_lanelet = lanelet
                else:
                    current_merged_lanelet = Lanelet.merge_lanelets(
                        current_merged_lanelet, lanelet
                    )

        goal_lanelet = self.lanelet_network.find_lanelet_by_id(
            self.route.list_ids_lanelets[-1]
        )
        if current_merged_lanelet is not None:
            current_merged_lanelet = Lanelet.merge_lanelets(
                current_merged_lanelet, goal_lanelet
            )
        else:
            current_merged_lanelet = goal_lanelet

        # Append successor of the goal to ensure that the goal state is not out of the projection domain
        # goal_lanelet = self.lanelet_network.find_lanelet_by_id(self.route.route[-1])
        # successors_of_goal = goal_lanelet.successor
        # if successors_of_goal is not None and len(successors_of_goal) != 0:
        #     successor_lanelet = self.lanelet_network.find_lanelet_by_id(successors_of_goal[0])
        #     current_merged_lanelet = Lanelet.merge_lanelets(current_merged_lanelet, successor_lanelet)

        self.merged_route_lanelets.append(current_merged_lanelet)

        return [
            self.create_coordinate_system_from_polyline(merged_lanelet.center_vertices)
            for merged_lanelet in self.merged_route_lanelets
        ]

    @classmethod
    def create_coordinate_system_from_polyline(
        cls,
        polyline: np.ndarray,
        return_resampled_polyline: bool = False,
    ) -> Tuple[pycrccosy.CurvilinearCoordinateSystem, np.ndarray]:
        """
        create CurvilinearCoordinateSystem from resampled polyline.

        :param polyline: np.ndarray[float[int>1, 2]] polyline
        :return:
            CurvilinearCoordinateSystem
            Resampled Polyline if flag return_resampled_polyline is set.
        """
        # version 2020, # version 06.2021
        assert polyline.ndim == 2

        if polyline.ndim == 2 and polyline.shape[0] == 2:
            step = 0.5
        else:
            step = min(compute_polyline_length(polyline) / 10.0, 0.5)
        resampled_polyline = resample_polyline(polyline, step=step)

        # remove consecutive duplicates, because they hurt runtime and may cause out of domain errors
        # disabled, leads to crash for nuplan dataset
        #resampled_polyline_shifted = np.roll(resampled_polyline, 1, axis=0)
        # slice all x,y points, where the next point is np.isclose to previous one
        #index_slice = np.any(np.invert(np.isclose(resampled_polyline, resampled_polyline_shifted)), axis=1)
        # sort out duplicated points
        #resampled_polyline = resampled_polyline[index_slice]

        if resampled_polyline.shape[0] <= 3:
            logger.warning(
                f"resampled polyline for CurvilinearCoSy has less then three points."
                f"all resampled points are  on the edge of the domain. resampled_polyline: {resampled_polyline} polyline {polyline}"
            )
        ccosy = pycrccosy.CurvilinearCoordinateSystem(resampled_polyline)
        if return_resampled_polyline:
            return ccosy, resampled_polyline
        else:
            return ccosy

    def _get_safe_curvilinear_coords(
        self, ccosy, position: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        # version 2020

        try:
            rel_pos_to_domain = 0
            long_lat_distance = self._get_curvilinear_coords(ccosy, position)
        except ValueError:
            long_lat_distance, rel_pos_to_domain = self._project_out_of_domain(
                ccosy, position
            )

        return np.array(long_lat_distance), rel_pos_to_domain

    def _project_out_of_domain(
        self, ccosy, position: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """for projection out of domain.
        Extend start and end of ccosy linearily with tangent"""
        # version 2020

        eps = 0.0001
        curvi_coords_of_projection_domain = np.array(
            ccosy.curvilinear_projection_domain()
        )

        longitudinal_min, normal_min = (
            np.min(curvi_coords_of_projection_domain, axis=0) + eps
        )
        longitudinal_max, normal_max = (
            np.max(curvi_coords_of_projection_domain, axis=0) - eps
        )
        normal_center = (normal_min + normal_max) / 2
        bounding_points = np.array(
            [
                ccosy.convert_to_cartesian_coords(longitudinal_min, normal_center),
                ccosy.convert_to_cartesian_coords(longitudinal_max, normal_center),
            ]
        )
        rel_positions = position - np.array(
            [bounding_point for bounding_point in bounding_points]
        )
        distances = np.linalg.norm(rel_positions, axis=1)

        if distances[0] < distances[1]:
            # Nearer to the first bounding point
            rel_pos_to_domain = -1
            long_dist = longitudinal_min + np.dot(
                ccosy.tangent(longitudinal_min), rel_positions[0]
            )
            lat_dist = normal_center + np.dot(
                ccosy.normal(longitudinal_min), rel_positions[0]
            )
        else:
            # Nearer to the last bounding point
            rel_pos_to_domain = 1
            long_dist = longitudinal_max + np.dot(
                ccosy.tangent(longitudinal_max), rel_positions[1]
            )
            lat_dist = normal_center + np.dot(
                ccosy.normal(longitudinal_max), rel_positions[1]
            )

        return np.array([long_dist, lat_dist]), rel_pos_to_domain

    def _get_curvilinear_coords(self, ccosy, position: np.ndarray) -> np.ndarray:
        # version 06.2021
        return ccosy.convert_to_curvilinear_coords(position[0], position[1])

    def _get_curvilinear_coords_over_lanelet(self, lanelet: Lanelet, position):
        # version 2020
        current_ccosy = self.create_coordinate_system_from_polyline(
            lanelet.center_vertices
        )

        return self._get_curvilinear_coords(current_ccosy, position)

    @staticmethod
    def _get_goal_face_points(goal_shape: Polygon):
        """
        Extracts the middle points of each face of the goal region
        NOTE in pathological examples, this can still result in points outside the coordinate system
        however for the points on both ends of the lanelet, they coincide with the center vertices,
        which is what the curvilinear coordinate system is based on
        NOTE if the goal areas edges are not all intersecting with any lanelet in the ccosy, this operation will fail
        :param goal_shape: shape of the goal area
        :return: tuples of x,y coordinates of the middle points of each face of the goal region
        """
        # version 2020 # version 06.2021
        assert isinstance(goal_shape, Polygon), (
            f"Only single Polygon is supported, but {type(goal_shape)} was given,"
            f" Use a planning problem with contiguous goal region"
        )

        goal_coords = [np.array(x) for x in zip(*goal_shape.exterior.coords.xy)]

        # round the same precision as is done within the commonroad xml files
        goal_coords = [
            np.round((a + b) / 2, 6) for a, b in zip(goal_coords, goal_coords[1:])
        ]
        return goal_coords

    def _get_goal_polygon(self, goal: GoalRegion) -> Polygon:
        """
        Get the goal position as Polygon
        :param goal: the goal given as a GoalRegion
        :return: Polygon of the goal position
        """
        # version 2020 # version 06.2021

        def get_polygon_list_from_shapegroup(
            shapegroup: cr_shape.ShapeGroup,
        ) -> List[Polygon]:
            """
            Converts cr_shape.ShapeGroup to list of Polygons
            :param shapegroup: the ShapeGroup to be converted
            :return: The list of the polygons
            """

            polygon_list = []
            for shape in shapegroup.shapes:
                if isinstance(shape, cr_shape.ShapeGroup):
                    polygon_list.append(get_polygon_list_from_shapegroup(shape))
                elif isinstance(shape, (cr_shape.Rectangle, cr_shape.Polygon)):
                    polygon_list.append(shape.shapely_object)
                else:
                    raise ValueError(
                        f"Shape can't be converted to Shapely Polygon: {shape}"
                    )
            return polygon_list

        def merge_polygons(polygons_to_merge):
            return unary_union(
                [
                    geom if geom.is_valid else geom.buffer(0)
                    for geom in polygons_to_merge
                ]
            )

        polygons = [Polygon()]
        for goal_state in goal.state_list:
            if hasattr(goal_state, "position"):
                if isinstance(goal_state.position, cr_shape.ShapeGroup):
                    polygons.extend(
                        get_polygon_list_from_shapegroup(goal_state.position)
                    )
                elif isinstance(
                    goal_state.position, (cr_shape.Rectangle, cr_shape.Polygon)
                ):
                    polygons.append(goal_state.position.shapely_object)
                else:
                    raise NotImplementedError(
                        f"Goal position not supported yet, "
                        f"only ShapeGroup, Rectangle or Polygon shapes can be used, "
                        f"the given shape was: {type(goal_state.position)}"
                    )

        merged_polygon = merge_polygons(polygons)
        return merged_polygon

    def get_position_curvi_coords(self, ego_vehicle_state_position: np.ndarray):
        # version 2020
        for cosy_idx, curvi_cosy in enumerate(self.ccosy_list):

            ego_curvi_coords, rel_pos_to_domain = self._get_safe_curvilinear_coords(
                curvi_cosy, ego_vehicle_state_position
            )

            is_last_section = cosy_idx == self.num_of_lane_changes - 1
            if rel_pos_to_domain == 1 and not is_last_section:
                continue

            return ego_curvi_coords, cosy_idx

        raise ValueError("Unable to project the ego vehicle on the global cosy")

    def get_long_lat_distance_to_goal(
        self, ego_vehicle_state_position: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get the longitudinal and latitudinal distance from the ego vehicle to the goal,
        measured in the frenet cos of the Navigator lanelet
        :param ego_vehicle_state_position: position of the ego vehicle
        :return:
            longitudinal distance of closest point on ref_path to goal
            latitudinal distance if in domain,
        """
        # version 2020
        # If the route is survival, then return zero
        if self.route.type == RouteType.SURVIVAL:
            return 0.0, 0.0

        ego_curvi_coords, cosy_idx = self.get_position_curvi_coords(
            ego_vehicle_state_position
        )

        is_last_section = cosy_idx == self.num_of_lane_changes - 1

        if is_last_section:
            relative_distances = self.goal_curvi_face_coords - ego_curvi_coords
            min_distance = np.min(relative_distances, axis=0)
            max_distance = np.max(relative_distances, axis=0)

            (min_distance_long, min_distance_lat) = np.maximum(
                np.minimum(0.0, max_distance), min_distance
            )
        else:
            min_distance_long = (
                self.merged_section_length_list[cosy_idx] - ego_curvi_coords[0]
            )
            current_section_idx = cosy_idx + 1
            while current_section_idx != self.num_of_lane_changes - 1:
                min_distance_long += self.merged_section_length_list[
                    current_section_idx
                ]
                current_section_idx += 1

            relative_lat_distances = (
                self.goal_curvi_face_coords[:, 1] - ego_curvi_coords[1]
            )
            min_distance = np.min(relative_lat_distances, axis=0)
            max_distance = np.max(relative_lat_distances, axis=0)

            min_distance_long += self.goal_min_curvi_coords[0]
            min_distance_lat = np.maximum(np.minimum(0.0, max_distance), min_distance)

        return min_distance_long, min_distance_lat

    def _get_safe_distance_to_curvilinear2(
        self, id_curvilinear: Union[str, int], position_ego: np.ndarray
    ):
        """
        distance to curvilinear with fallback to closest point on reference path via kdtree
        :param id_curvilinear: unique string, lanelet_id or "reference_path"
        :param position_ego: np.ndarray[float[2, 1]],

        :return:
            p_curvilinear_closest[0]: float[0, ccosy.length()], closest longitudinal coordinate to position_ego
            distance: float[0, ccosy.length()], distance to closest point on reference path
            ccosy: pycrccosy.CurvilinearCoordinateSystem
            indomain: float, 0.0, if position_ego in domain, else 1
        """
        # version 06.2021
        if id_curvilinear not in self.lane_ccosy_kd_ref_dict:
            # create if this is the first time of inferencing this lanelet_id
            raise BaseException(
                f" error in _get_safe_distance_to_curvilinear2: id_curvilinear not in self.lane_ccosy_kd_ref_dict {id_curvilinear} {self.lane_ccosy_kd_ref_dict.keys()}"
            )

        ccosy, kdtree, resampled_polyline = self.lane_ccosy_kd_ref_dict[id_curvilinear]

        indomain = 1.0
        try:
            p_curvilinear_closest = ccosy.convert_to_curvilinear_coords(
                position_ego[0], position_ego[1]
            )
            distance = p_curvilinear_closest[1]
            indomain = 1.0

        except ValueError:
            try:
                # for projection out of domain
                # logger.debug(
                #     f"projection of ego vehicle {position_ego} to reference path of id {id_curvilinear} CCoSy is out of domain. "
                #     "defaulting to closest distance on reference path."
                # )
                # get index of 2D reference_path array to which position_ego is closest
                distance, idx = kdtree.query(position_ego)
                closest_point = resampled_polyline[idx]

                p_curvilinear_closest = ccosy.convert_to_curvilinear_coords(
                    closest_point[0], closest_point[1]
                )
                # Done, but distance is yet not in curvilinear direction.
                # return indomain as angle between ego and point on ccosy to the curvilinear CoSy.
                if np.isclose(distance, 0, atol=1e-3):
                    indomain = 1
                else:
                    tangent = self.safe_tangent(ccosy, p_curvilinear_closest[0])
                    tangent = tangent / np.linalg.norm(tangent)
                    to_closest = position_ego - closest_point
                    # projected on the tangent of the closest curvilinear, take portion which is normal to
                    lateral_distance_to_tangent = np.cross(tangent, to_closest)
                    # add sign to the distance,
                    distance = math.copysign(distance, lateral_distance_to_tangent)

                    # indomain is now cos of the angle of the ego towards point w. tangent on ccosy
                    indomain = lateral_distance_to_tangent / distance
            except Exception as ex:
                raise BaseException(
                    f" error in _get_safe_distance_to_curvilinear2: failed in id_curvilinear {id_curvilinear} closest_point {closest_point} idx{idx} resampled_polyline {resampled_polyline}. \n Exception: {ex}"
                )
        except Exception as ex:
            raise BaseException(
                f" error in _get_safe_distance_to_curvilinear2: failed in id_curvilinear {id_curvilinear} resampled_polyline {resampled_polyline}. \n Exception: {ex}"
            )

        return p_curvilinear_closest[0], distance, ccosy, indomain

    def get_lane_change_distance(
        self, state: State, active_lanelets: List[int] = None
    ) -> float:
        # version 2020
        """
        get distance in the frenet cos of the reference path to the end of the current lanelet


        """
        # If the route is survival, then return zero
        if self.route.type == RouteType.SURVIVAL:
            return 0.0
        if active_lanelets is None:
            active_lanelets = self.scenario.lanelet_network.find_lanelet_by_position(
                [state.position]
            )[0]

        current_lanelet_ids_on_route = [
            current_lanelet_id
            for current_lanelet_id in active_lanelets
            if current_lanelet_id in self.sectionized_environment_set
        ]
        # The state is not on the route, instant lane change is required
        if len(current_lanelet_ids_on_route) == 0:
            return 0.0

        sorted_current_lanelet_ids_on_route = Navigator.sorted_lanelet_ids(
            current_lanelet_ids_on_route,
            state.orientation,
            state.position,
            self.scenario,
        )

        # The most likely current lanelet id by considering the orientation of the state
        current_lanelet_id = sorted_current_lanelet_ids_on_route[0]

        distance_until_lane_change = 0.0
        route_successors = {current_lanelet_id}
        while len(route_successors) != 0:
            # Add the length of the current lane
            current_lanelet_id = route_successors.pop()
            current_lanelet = self.lanelet_network.find_lanelet_by_id(
                current_lanelet_id
            )
            try:
                if distance_until_lane_change == 0.0:
                    # Calculate the remaining distance in this lanelet
                    current_distance = self._get_curvilinear_coords_over_lanelet(
                        current_lanelet, state.position
                    )
                    current_distance_long = current_distance[0]
                    distance_until_lane_change = (
                        current_lanelet.distance[-1] - current_distance_long
                    )
                else:
                    distance_until_lane_change += current_lanelet.distance[-1]

            except ValueError:
                pass

            successors_set = set(current_lanelet.successor)
            route_successors = successors_set.intersection(
                self.sectionized_environment_set
            )

        return distance_until_lane_change

    @classmethod
    def _evaluate_orientation_observations(
        cls,
        state=State,
        observation_cos: CosyVehicleObservation = CosyVehicleObservation.AUTOMATIC,
    ) -> float:
        """get orientation of the ego cos for observations.

        CosyVehicleObservation.VEHICLEFRAME:
            always rotated in ego orienation (fallback velocity heading)
        CosyVehicleObservation.LOCALCARTESIAN:
            never rotated / orienation in global east /
        CosyVehicleObservation.AUTOMATIC:
            VEHICLEFRAME if state has orientation, else LOCALCARTESIAN

        """
        # version 06.2021
        if observation_cos is None:
            observation_cos = cls.CosyVehicleObservation.AUTOMATIC

        if observation_cos == cls.CosyVehicleObservation.LOCALCARTESIAN:
            # orienation not considered
            return 0.0
        elif hasattr(state, "orientation") and observation_cos in [
            cls.CosyVehicleObservation.AUTOMATIC,
            cls.CosyVehicleObservation.VEHICLEFRAME,
        ]:
            return float(state.orientation)
        else:
            # for PM model get orientation from velocity
            if observation_cos == cls.CosyVehicleObservation.VEHICLEFRAME:
                # get orientation from velocity
                if hasattr(state, "velocity_y") and hasattr(state, "velocity"):
                    return np.arctan2(state.velocity_y, state.velocity)
                else:
                    raise BaseException(
                        "State has no orienation or global x & y orienations. Cant compute vehicle frame CoSy."
                    )
            else:
                return 0.0

    def get_waypoints_of_reference_path(
        self,
        state,
        distances_ref_path: Set[float],
        observation_cos: CosyVehicleObservation = CosyVehicleObservation.AUTOMATIC,
    ):
        """
        computes the vectors in observation_cos to the waypoints in the reference path ccosy.
        waypoints are on the reference path, at lonitudinal positions: distances_ref_path,
        compared to the closest position of the ego on the reference path
        returning the vectors to the points on reference path in ego local_cosy and relative orientaion.

        :param distances_ref_path: Set of Waypoints at distance d to be computed. Include int(0)
        :param state: state of the ego vehicle
        :param observation_cos: string, in which vehicle coordinate system observation should be made
        :return:
            np.ndarray: vectors to points, shape: (len(distances_ref_path), 2)
            np.ndarray: orientations at points, shape: (len(distances_ref_path), )

        minimal example:
        Navigator.get_waypoints_of_reference_path(
            state=EgoVehicle.State,
            distances_ref_path=[-1000, 0, 1000],
            observation_cos=CosyVehicleObservation.AUTOMATIC,
        )
        """
        # version 06.2021
        position_ego = state.position

        orientation_vehicle = self._evaluate_orientation_observations(
            state, observation_cos
        )

        # self.route.clcs is pycrccosy CurvilinearCoordinateSystem of route.ref_path
        # position closet on reference path in clcs: p_curvilinear = [long_clcs, lat_clcs]
        p_curvilinear, _, ccosy, _ = self._get_safe_distance_to_curvilinear2(
            id_curvilinear="reference_path", position_ego=position_ego
        )
        vectors_to_ref, orientations_ref = self.get_points_on_ccosy(
            ccosy, p_curvilinear, distances_ref_path, orientation_vehicle, position_ego
        )
        return (vectors_to_ref, orientations_ref)

    def get_longlat_togoal_on_reference_path(self, state) -> float:
        """
        returns the distance  [m] to the closest point on the reference path,
        and the distance to the end of the curvilinear cosy

        :param position_ego: position of ego, array of shape (2,)
        :return:
            float, longitudinal distance to point of reference path which is closest to goal
            float, distance to reference path
            float, indomain:
                1 if indomain (distance to reference path orthogonal to ccosy),
                0-1 if angle towards domain not orthodgonal

        minimal example:
        float, float = get_longlat_togoal_on_reference_path(np.array([0,10]))
        """
        # version 06.2021

        (
            p_curvilinear_long,
            distance,
            ccosy,
            indomain,
        ) = self._get_safe_distance_to_curvilinear2(
            id_curvilinear="reference_path", position_ego=state.position
        )

        if p_curvilinear_long < self.goal_min_curvi_coords_ref:
            min_distance_to_goal_long_on_ref = (
                self.goal_min_curvi_coords_ref - p_curvilinear_long
            )
        elif p_curvilinear_long > self.goal_max_curvi_coord_ref:
            min_distance_to_goal_long_on_ref = (
                self.goal_max_curvi_coord_ref - p_curvilinear_long
            )
        else:
            min_distance_to_goal_long_on_ref = 0.0

        return min_distance_to_goal_long_on_ref, distance, indomain

    @staticmethod
    def safe_convert_to_cartesian_coords(
        ccosy: pycrccosy.CurvilinearCoordinateSystem,
        long_lat_eval: Tuple[float, float],
        eps: float = 0.001,
        max_lat: float = 20.0,
        length_ccosy: Union[float, None] = None,
    ) -> Tuple[float, float]:
        """evaluate from ccosy to cartesian with respect to boundaries

        :param ccosy: Curvilinear System
        :param long_lat_eval: long and lat position in ccosy, for which cartesian shall be returned
        :optional:
            :param eps: nearest longitudinal distance allowed to ends of ccosy. default: 0.001m
            :param max_lat: nearest lateral distance allowed to center of ccosy.
                default: 20m == default max set during ccosy creation
            :param length_ccosy: length of ccosy, if held in memory

        :return: corresponding cartesian position
        """
        try:
            if length_ccosy is None:
                length_ccosy = ccosy.length()
            long_clcs_eval = min(max(long_lat_eval[0], eps), length_ccosy - eps)
            lat_clcs_eval = min(max(long_lat_eval[1], -max_lat + eps), max_lat)
            return ccosy.convert_to_cartesian_coords(long_clcs_eval, lat_clcs_eval)
        except ValueError as ex:
            raise BaseException(
                f"try to evaluate long: {long_clcs_eval} lat: {lat_clcs_eval} on ccosy but got ValueError {ex}"
            )

    @staticmethod
    def safe_tangent(
        ccosy: pycrccosy.CurvilinearCoordinateSystem,
        long_eval: float,
        eps: float = 0.001,
        length_ccosy: Union[float, None] = None,
    ) -> Tuple[float, float]:
        """evaluate from tanget at a position in the ccosy with respect to boundaries

        :param ccosy: Curvilinear System
        :param long_eval: long  position in ccosy, for which tangent shall be returned
        :optional:
            :param eps: nearest longitudinal distance allowed to ends of ccosy. default: 0.001m
            :param length_ccosy:  length of ccosy, if held in memory, default: None

        :return: corresponding cartesian position
        """
        if length_ccosy is None:
            length_ccosy = ccosy.length()
        long_clcs_eval = min(max(long_eval, eps), length_ccosy - eps)
        return ccosy.tangent(long_clcs_eval)

    @classmethod
    def get_points_on_ccosy(
        cls,
        ccosy: pycrccosy.CurvilinearCoordinateSystem,
        p_curvilinear: float,
        distances_ref_path: set,
        orientation_ego: float,
        position_ego: np.ndarray,
    ):
        """
        computes the vectors in ego/local_cosy to the points for (p_curvilinear+distances_ref_path) in ccosy.
        returning the vectors to the points on reference path in ego local_cosy and relative orientaion.

        :raises ValueError: if (p_curvilinear, 0) not in ccosy domain

        :param distances_ref_path: Set of Waypoints at distance d to be computed. Include int(0)
        :param state: state of the ego vehicle
        :return:
            np.ndarray: vectors to points, shape: (len(distances_ref_path), 2)
            np.ndarray: orientations at points, shape: (len(distances_ref_path), )

        minimal example:
        positions, lane_orienations = Navigator.get_points_on_ccosy(
            ccosy = pycrccosy.CurvilinearCoordinateSystem(),
            p_curvilinear = 5.1,
            distances_ref_path = [-np.inf, -5, 0, 10, 50],
            orientation_ego = np.pi,
            position_ego = np.array([0, 0])
        )
        """
        # version 06.2021
        vectors_to_ref = []
        v_ref_tangent = []

        distances_ref_path = sorted(distances_ref_path)

        length_ccosy = ccosy.length()

        projected_point = None
        projected_tangent = None

        # get observations: cartesian distance and orientation of reference waypoints in vehicle_cos
        for distance_ref in distances_ref_path:
            # distance in positive direction of the clcs
            long_clcs_eval = p_curvilinear + distance_ref
            # make valid in ccosy

            try:
                # try to get true distance
                projected_point = cls.safe_convert_to_cartesian_coords(
                    ccosy,
                    long_lat_eval=[long_clcs_eval, 0.0],
                    length_ccosy=length_ccosy,
                )
                projected_tangent = cls.safe_tangent(ccosy, long_clcs_eval)
            except ValueError as ex:
                logger.error(
                    f"coordinate at long {long_clcs_eval} {length_ccosy} has exception {ex}. padding with last value: {projected_point}"
                )
                # use padding with last working value.
                if projected_point is None:
                    projected_point = cls.safe_convert_to_cartesian_coords(
                        ccosy,
                        long_lat_eval=[p_curvilinear, 0.0],
                        length_ccosy=length_ccosy,
                    )
                    projected_tangent = cls.safe_tangent(ccosy, p_curvilinear)
            finally:
                vectors_to_ref.append(projected_point)
                v_ref_tangent.append(projected_tangent)

        # positions and orientations of waypoints collected, for lower variance project now on vehicle cos
        vectors_to_ref = np.asarray(vectors_to_ref) - position_ego  # relative vector
        v_ref_tangent = np.asarray(
            v_ref_tangent
        )  # unit length tangent vector at dist i of ref_path
        # compute relative orientations at the ref_path points
        orientations_ref = (
            np.arctan2(v_ref_tangent[:, 1], v_ref_tangent[:, 0]) - orientation_ego
        )
        # rotate matrix arond orientation, so
        vectors_to_ref, _ = cls._rotate_2darray_by_orientation(
            orient=orientation_ego, twodim_array=vectors_to_ref
        )
        return vectors_to_ref, orientations_ref

    @staticmethod
    def _rotate_2darray_by_orientation(
        orient: float,
        twodim_array: np.ndarray,
        rot_matrix: Union[None, np.matrix] = None,
    ) -> np.ndarray:
        """
        rotates "list of 2d poins" array by orientation

        :param orient: float [-pi, pi]
        :param twodim_array: np.ndarray[float[int, 2]]
        :param rot_matrix: None or np.matrix[2, 2]
        :return:
            np.ndarray: rotated twodim_array shape (n,2)
            np.matrix: np.ndarray shape (2,2)
        """

        # version 06.2021
        assert (
            twodim_array.shape[0] > 0 and twodim_array.shape[1] == 2
        ), f"imput twodim_array has wrong shape {twodim_array.shape}"
        if float(orient) == 0.0:
            return twodim_array, np.asarray([[1, 0], [0, 1]])
        elif rot_matrix is None:
            c = np.cos(orient)
            s = np.sin(orient)
            rot_matrix = np.array([[c, s], [-s, c]]).T
        else:
            assert rot_matrix.shape == (2, 2)

        # return rotated array and rotation matrix for further usage
        return (rot_matrix @ twodim_array.T).T, rot_matrix

    def get_referencepath_multilanelets_waypoints(
        self,
        state: State,
        distances_per_lanelet: Set[float],
        lanelets_id_rel: Set[int] = (-1, 0, 1, 2),
        observation_cos: CosyVehicleObservation = CosyVehicleObservation.AUTOMATIC,
    ):
        """
        returns the vectors and orientations for waypoints on a set of selected lanelets (centerlines).
        include lanelets which are at lanelets_id_rel relative to current closest lanelet to ego.
        (0: current, -1 predecessor in planned route, +1 next in planned route etc.)
        compute waypoints at positions: distances_per_lanelet for each lanelet.

        :param state: ego state
        :lanelets_id_rel: List[int]
        lanelets_id_rel: List[int] = [0, 1]
        """
        # version 06.2021

        computed_waypoints = np.zeros(
            (len(lanelets_id_rel), len(distances_per_lanelet), 2)
        )
        computed_orientations = np.zeros(
            (len(lanelets_id_rel), len(distances_per_lanelet))
        )

        position_ego = state.position

        orientation_vehicle = self._evaluate_orientation_observations(
            state, observation_cos
        )

        distance_lanelets = []
        # self.route.clcs is pycrccosy CurvilinearCoordinateSystem of route.ref_path
        # position closet on reference path in clcs: p_curvilinear = [long_clcs, lat_clcs]
        for id_lanelet in self.route.list_ids_lanelets:
            if id_lanelet not in self.lane_ccosy_kd_ref_dict:
                raise BaseException(
                    f" error in get_referencepath_multilanelets_waypoints: id_lanelet not in self.lane_ccosy_kd_ref_dict {id_lanelet} {self.lane_ccosy_kd_ref_dict.keys()}"
                )

            (
                p_curvilinear_long,
                distance,
                ccosy,
                _,
            ) = self._get_safe_distance_to_curvilinear2(
                id_curvilinear=id_lanelet, position_ego=position_ego
            )
            distance_lanelets.append((id_lanelet, p_curvilinear_long, distance, ccosy))

        if not distance_lanelets:
            raise BaseException(
                f"ego vehicle with position {position_ego}"
                f"has no closest lanelet in lanelet ids: {self.route.list_ids_lanelets}"
            )

        distance_lanelets.sort(
            key=lambda tup: abs(tup[2])
        )  # sorts in place by distance

        # logger.error(distance_lanelets)

        idx_closest_lanelet = self.route.list_ids_lanelets.index(
            distance_lanelets[0][0]
        )
        for count, rel_id in enumerate(lanelets_id_rel):
            # rel_id is 0 for closest, +1 for successor of closest, +-1 for predecessor of closest
            idx_observe = min(
                max(rel_id + idx_closest_lanelet, 0),
                len(self.route.list_ids_lanelets) - 1,
            )  # padding to first / last if this position is out of bound

            lanelet_id_observe = self.route.list_ids_lanelets[idx_observe]

            id_lane, p_curvi_long, distance_lat, ccosy = next(
                (id_lane, p_curvi_long, distance_lat, ccosy)
                for (id_lane, p_curvi_long, distance_lat, ccosy) in distance_lanelets
                if id_lane == lanelet_id_observe
            )

            # make obervation, where the predecessor lanelet is.

            vectors_to_ref, orientations_ref = self.get_points_on_ccosy(
                ccosy,
                p_curvi_long,
                distances_per_lanelet,
                orientation_vehicle,
                position_ego,
            )
            # logger.debug(
            #     f"observe points {vectors_to_ref} and orients {orientations_ref} for lanelet {lanelet_id_observe}, "
            #     f"which is at position {rel_id} to closest lanelet {distance_lanelets[0][0]} "
            #     f"at ego position {position_ego}"
            # )
            computed_waypoints[count, :, :] = vectors_to_ref
            computed_orientations[count, :] = orientations_ref

        return (computed_waypoints, computed_orientations)

    @staticmethod
    def sorted_lanelet_ids(
        lanelet_ids: List[int],
        orientation: float,
        position: np.ndarray,
        scenario: Scenario,
    ) -> List[int]:
        """
        return the lanelets sorted by relative orientation to the position and orientation given
        """
        # version 2020
        if len(lanelet_ids) <= 1:
            return lanelet_ids
        else:
            lanelet_id_list = np.array(lanelet_ids)

            def get_lanelet_relative_orientation(lanelet_id):
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                lanelet_orientation = Navigator.lanelet_orientation_at_position(
                    lanelet, position
                )
                return np.abs(
                    Navigator.relative_orientation(lanelet_orientation, orientation)
                )

            orientation_differences = np.array(
                list(map(get_lanelet_relative_orientation, lanelet_id_list))
            )
            sorted_indices = np.argsort(orientation_differences)
            return list(lanelet_id_list[sorted_indices])

    @staticmethod
    def lanelet_orientation_at_position(lanelet: Lanelet, position: np.ndarray):
        """
        Approximates the lanelet orientation with the two closest point to the given state
        # TODO optimize more for speed

        :param lanelet: Lanelet on which the orientation at the given state should be calculated
        :param position: Position where the lanelet's orientation should be calculated
        :return: An orientation in interval [-pi,pi]
        """
        # version 2020

        center_vertices = lanelet.center_vertices

        position_diff = []
        for idx in range(len(center_vertices) - 1):
            vertex1 = center_vertices[idx]
            position_diff.append(np.linalg.norm(position - vertex1))

        closest_vertex_index = position_diff.index(min(position_diff))

        vertex1 = center_vertices[closest_vertex_index, :]
        vertex2 = center_vertices[closest_vertex_index + 1, :]
        direction_vector = vertex2 - vertex1
        return np.arctan2(direction_vector[1], direction_vector[0])

    @staticmethod
    def relative_orientation(from_angle1_in_rad, to_angle2_in_rad):
        # version 2020
        phi = (to_angle2_in_rad - from_angle1_in_rad) % TWO_PI
        if phi > np.pi:
            phi -= TWO_PI

        return phi
