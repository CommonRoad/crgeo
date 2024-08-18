import warnings
from typing import Dict, Optional

import networkx as nx
import numpy as np
import math
from commonroad.geometry.shape import Shape, ShapeGroup
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import ScenarioID

from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.dataset.extraction.traffic.feature_computers import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation

EPS = 1e-1


class GoalAlignmentComputer(BaseFeatureComputer[VFeatureParams]):
    @classproperty
    def allow_nan_values(cls) -> bool:
        return True

    def __init__(
        self,
        include_goal_distance_longitudinal: bool = True,
        include_goal_distance_lateral: bool = True,
        include_goal_distance: bool = True,
        include_lane_changes_required: bool = True,
        logarithmic: bool = True,
        closeness_transform: bool = False,
        closeness_transform_threshold: float = 100.0
    ) -> None:
        if not any((
            include_goal_distance_longitudinal,
            include_goal_distance_lateral,
            include_goal_distance,
            include_lane_changes_required
        )):
            raise ValueError("GoalAlignmentComputer doesn't include any features")

        self._include_goal_distance_longitudinal = include_goal_distance_longitudinal
        self._include_goal_distance_lateral = include_goal_distance_lateral
        self._include_goal_distance = include_goal_distance
        self._include_lane_changes_required = include_lane_changes_required
        self._logarithmic = logarithmic
        self._closeness_transform = closeness_transform
        self._closeness_transform_threshold = closeness_transform_threshold
        assert not logarithmic and closeness_transform
        self._lanelet_network: Optional[LaneletNetwork] = None
        self._scenario_id: Optional[ScenarioID] = None
        self._undefined_features = self._return_undefined_features()
        self._lane_changes_required_cache: Dict[(int, int), (int, int)] = {}
        super().__init__()

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        if not params.is_ego_vehicle or params.ego_route is None:
            return self._undefined_features
        
        self._reset(simulation)

        features: FeatureDict = {}

        ego_state = params.state
        position = ego_state.position
        orientation = ego_state.orientation
        lanelet_graph = simulation.lanelet_graph

        if self._include_goal_distance_lateral or self._include_goal_distance_longitudinal:
            distance_goal_long, distance_goal_lat = params.ego_route.navigator.get_long_lat_distance_to_goal(position)

            if self._include_goal_distance_longitudinal:
                # the logarithmic option acts as a sort of normaliziation to avoid large feature values
                if self._logarithmic:
                    if distance_goal_long < 0:
                        features[V_Feature.GoalDistanceLongitudinal.value] = -np.log(-distance_goal_long + 1)
                    else:
                        features[V_Feature.GoalDistanceLongitudinal.value] = np.log(distance_goal_long + 1)
                elif self._closeness_transform:
                    features[V_Feature.GoalDistanceLongitudinal.value] = GoalAlignmentComputer._closeness_transform_fn(distance_goal_long, self._closeness_transform_threshold)
                else:
                    features[V_Feature.GoalDistanceLongitudinal.value] = distance_goal_long / 10.0
            
            if self._include_goal_distance_lateral:
                if self._logarithmic:
                    if distance_goal_lat < 0:
                        features[V_Feature.GoalDistanceLateral.value] = -np.log(-distance_goal_lat + 1)
                    else:
                        features[V_Feature.GoalDistanceLateral.value] = np.log(distance_goal_lat + 1)
                else:
                    features[V_Feature.GoalDistanceLateral.value] = distance_goal_lat / 10.0

        if self._include_goal_distance:
            distance, heading_error = GoalAlignmentComputer._get_goal_distance_and_orientation(position, orientation, params.ego_route.goal_region)
            if self._logarithmic:
                features[V_Feature.GoalDistance.value] = np.log(distance + EPS)
            if self._closeness_transform:
                features[V_Feature.GoalDistance.value] = GoalAlignmentComputer._closeness_transform_fn(distance, self._closeness_transform_threshold)
            else:
                features[V_Feature.GoalDistance.value] = distance / 10.0
            features["goal_heading_error"] = heading_error

        if self._include_lane_changes_required:
            ego_lanelet = simulation.get_obstacle_lanelet(params.obstacle)

            if ego_lanelet is None:
                shortest_path_adjacent, shortest_path_direction = np.nan, np.nan
            elif (ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id) in self._lane_changes_required_cache:
                shortest_path_adjacent, shortest_path_direction = self._lane_changes_required_cache[(
                    ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id)]
            else:
                found_shortest_path: bool = True
                try:
                    shortest_path = self._shortest_paths[ego_lanelet.lanelet_id][params.ego_route.goal_lanelet.lanelet_id]
                except KeyError:
                    found_shortest_path = False

                try:
                    if found_shortest_path:
                        shortest_path_successors = 0
                        shortest_path_direction = -1
                        for i in range(len(shortest_path) - 1):
                            edge = (shortest_path[i], shortest_path[i + 1])
                            if edge is None:
                                # TODO: What?
                                found_shortest_path = False
                                break
                            lanelet_edge_type = LaneletEdgeType(
                                lanelet_graph.get_edge_data(
                                    edge[0], edge[1])['lanelet_edge_type'])
                            if lanelet_edge_type is None:
                                # TODO: What?
                                found_shortest_path = False
                                break
                            if lanelet_edge_type in {LaneletEdgeType.SUCCESSOR, LaneletEdgeType.PREDECESSOR}:
                                shortest_path_successors += 1
                            elif lanelet_edge_type in {LaneletEdgeType.ADJACENT_RIGHT, LaneletEdgeType.DIAGONAL_RIGHT}:
                                shortest_path_direction = 1
                        shortest_path_adjacent = len(shortest_path) - 1 - shortest_path_successors
                        shortest_path_direction = shortest_path_direction if shortest_path_adjacent > 0 else 0
                        self._lane_changes_required_cache[(ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id)] = (
                            shortest_path_adjacent, shortest_path_direction)
                except TypeError:
                    found_shortest_path = False
                    # TODO ???

                if not found_shortest_path:
                    shortest_path_adjacent = 0.0  # np.nan
                    shortest_path_direction = 0.0  # np.nan

            features[V_Feature.LaneChangesRequired.value] = shortest_path_adjacent
            features[V_Feature.LaneChangeDirectionRequired.value] = shortest_path_direction

        return features

    def _return_undefined_features(self) -> FeatureDict:
        features: FeatureDict = {}

        if self._include_goal_distance_longitudinal:
            features[V_Feature.GoalDistanceLongitudinal.value] = 0.0
        if self._include_goal_distance_lateral:
            features[V_Feature.GoalDistanceLateral.value] = 0.0
        if self._include_goal_distance:
            features[V_Feature.GoalDistance.value] = 0.0
            features["goal_heading_error"] = 0.0
        if self._include_lane_changes_required:
            features[V_Feature.LaneChangesRequired.value] = 0.0
        if self._include_lane_changes_required:
            features[V_Feature.LaneChangeDirectionRequired.value] = 0.0

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        scenario_id = simulation.current_scenario.scenario_id
        if self._scenario_id is None or scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._lanelet_network = simulation.current_scenario.lanelet_network
            self._shortest_paths = nx.shortest_path(simulation.lanelet_graph)
            self._lane_changes_required_cache = {}

    @staticmethod
    def _closeness_transform_fn(x: float, threshold: float) -> float:
        if not math.isfinite(x):
            return 0.0
        return 1 - min(x, threshold) / threshold

    @staticmethod
    def _get_goal_distance_and_orientation(position: np.array, orientation: float, goal: GoalRegion) -> float:
        """
        Calculates the Euclidean distance and the orientation difference of the current position to the goal.

        :param position: current position as an array [x, y]
        :param position_orientation: current orientation in radians or degrees
        :param goal: the goal of the current planning problem
        :return: Tuple containing Euclidean distance and orientation difference
        """
        if "position" not in goal.state_list[0].attributes:
            return 0., 0.

        goal_position_list = []
        goal_orientation = 0.  # Assuming goal orientation can be obtained or calculated similarly

        f_pos = goal.state_list[0].position
        if isinstance(f_pos, ShapeGroup):
            goal_position_list = np.array(
                [GoalAlignmentComputer._convert_shape_group_to_center(s.position) for s in goal.state_list])
            # Add code to calculate goal_orientation for ShapeGroup if applicable
        elif isinstance(f_pos, Shape):
            goal_position_list = np.array([s.position.center for s in goal.state_list])
            # Add code to calculate goal_orientation for Shape if applicable
        else:
            warnings.warn(f"Trying to calculate relative goal orientation but goal state position "
                        f"type ({type(f_pos)}) is not support, please set "
                        f"observe_distance_goal_euclidean = False or "
                        f"change state position type to one of the following: Polygon, Rectangle, Circle")
            return 0., 0.

        goal_position_mean = np.mean(goal_position_list, axis=0)
        euclidean_distance = np.linalg.norm(position - goal_position_mean)

        # Calculate the orientation difference
        orientation_difference = relative_orientation(goal_orientation, orientation)

        return euclidean_distance, orientation_difference

    @staticmethod
    def _convert_shape_group_to_center(shape_group: ShapeGroup):
        position_list = [shape.center for shape in shape_group.shapes]
        return np.mean(np.array(position_list), axis=0)
