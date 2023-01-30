import warnings
from typing import Dict, Optional, TYPE_CHECKING

import networkx as nx
import numpy as np
from commonroad.geometry.shape import Shape, ShapeGroup
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import ScenarioID

from crgeo.common.class_extensions.class_property_decorator import classproperty
from crgeo.dataset.extraction.road_network.types import LaneletEdgeType
from crgeo.dataset.extraction.traffic.feature_computers import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from crgeo.simulation.base_simulation import BaseSimulation

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
        logarithmic: bool = False
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
        self._lanelet_network: Optional[LaneletNetwork] = None
        self._scenario_id: Optional[ScenarioID] = None
        self._undefined_features = self._return_undefined_features()
        super().__init__()

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        if not params.is_ego_vehicle or params.ego_route is None:
            return self._undefined_features

        features: FeatureDict = {}

        ego_state = params.state
        position = ego_state.position
        lanelet_graph = simulation.lanelet_graph

        if self._include_goal_distance_lateral or self._include_goal_distance_longitudinal:
            distance_goal_long, distance_goal_lat = params.ego_route.navigator.get_long_lat_distance_to_goal(position)
            if self._include_goal_distance_longitudinal:
                features[V_Feature.GoalDistanceLongitudinal.value] = np.log(distance_goal_long + EPS) if self._logarithmic else distance_goal_long
            if self._include_goal_distance_lateral:
                features[V_Feature.GoalDistanceLateral.value] = np.log(distance_goal_lat + EPS) if self._logarithmic else distance_goal_lat

        if self._include_goal_distance:
            distance = GoalAlignmentComputer._get_goal_euclidean_distance(position, params.ego_route.goal_region)
            features[V_Feature.GoalDistance.value] = np.log(distance + EPS) if self._logarithmic else distance

        if self._include_lane_changes_required:
            ego_lanelet = simulation.get_obstacle_lanelet(params.obstacle)

            if ego_lanelet is None:
                shortest_path_adjacent, shortest_path_direction = np.nan, np.nan
            elif (ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id) in self._lane_changes_required_cache:
                shortest_path_adjacent, shortest_path_direction = self._lane_changes_required_cache[(ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id)]
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
                            lanelet_edge_type = LaneletEdgeType(lanelet_graph.get_edge_data(edge[0], edge[1])['lanelet_edge_type'])
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
                        self._lane_changes_required_cache[(ego_lanelet.lanelet_id, params.ego_route.goal_lanelet.lanelet_id)] = (shortest_path_adjacent, shortest_path_direction)
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
            self._lane_changes_required_cache: Dict[(int, int), (int, int)] = {}

    @staticmethod
    def _get_goal_euclidean_distance(position: np.array, goal: GoalRegion) -> float:
        """
        calculates the euclidean distance of the current position to the goal

        :param position: current position
        :param goal: the goal of the current planning problem
        :return euclidean distance
        """
        if "position" not in goal.state_list[0].attributes:
            return 0.

        else:
            f_pos = goal.state_list[0].position
            if isinstance(f_pos, ShapeGroup):
                goal_position_list = np.array(
                    [GoalAlignmentComputer._convert_shape_group_to_center(s.position) for s in goal.state_list])
            elif isinstance(f_pos, Shape):
                goal_position_list = np.array([s.position.center for s in goal.state_list])
            else:
                warnings.warn(f"Trying to calculate relative goal orientation but goal state position "
                              f"type ({type(f_pos)}) is not support, please set "
                              f"observe_distance_goal_euclidean = False or "
                              f"change state position type to one of the following: Polygon, Rectangle, Circle")
                return 0.
            goal_position_mean = np.mean(goal_position_list, axis=0)
            return np.linalg.norm(position - goal_position_mean)

    @staticmethod
    def _convert_shape_group_to_center(shape_group: ShapeGroup):
        position_list = [shape.center for shape in shape_group.shapes]
        return np.mean(np.array(position_list), axis=0)
