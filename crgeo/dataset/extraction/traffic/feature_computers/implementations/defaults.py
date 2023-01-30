from typing import Sequence

from crgeo.dataset.extraction.traffic.feature_computers.implementations.lanelet.lanelet_geometry_feature_computer import LaneletGeometryFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet.lanelet_connection_geometry_feature_computer import LaneletConnectionGeometryFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle.callables import ft_veh_state
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle.num_lanelet_assignments_feature_computer import NumLaneletAssignmentsFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle.vehicle_lanelet_connectivity_feature_computer import VehicleLaneletConnectivityComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle.vehicle_lanelet_pose_feature_computer import VehicleLaneletPoseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle.yaw_rate_feature_computer import YawRateFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import VehicleLaneletPoseEdgeFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle.callables import ft_rel_state_ego, ft_same_lanelet
from crgeo.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle.lanelet_distance_feature_computer import LaneletDistanceFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.types import L2LFeatureParams, LFeatureParams, T_FeatureComputer, V2VFeatureParams, \
    V2LFeatureParams, VFeatureParams


class DefaultFeatureComputers:

    @staticmethod
    def v():
        return [
            ft_veh_state,
            YawRateFeatureComputer(),
            VehicleLaneletPoseFeatureComputer(),
            VehicleLaneletConnectivityComputer(),
            NumLaneletAssignmentsFeatureComputer(),
            #GoalAlignmentComputer()
        ]

    @staticmethod
    def v2v():
        return [
            ft_same_lanelet,
            LaneletDistanceFeatureComputer(),
            ft_rel_state_ego
        ]

    @staticmethod
    def l():
        return [
            LaneletGeometryFeatureComputer(),
        ]

    @staticmethod
    def l2l():
        return [
            LaneletConnectionGeometryFeatureComputer(),
        ]

    @staticmethod
    def v2l():
        return [
            VehicleLaneletPoseEdgeFeatureComputer()
        ]

    @staticmethod
    def l2v():
        return []
