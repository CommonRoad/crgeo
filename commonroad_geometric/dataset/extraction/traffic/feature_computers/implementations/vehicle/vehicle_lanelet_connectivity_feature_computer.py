from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class VehicleLaneletConnectivityComputer(BaseFeatureComputer[VFeatureParams]):
    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        lanelet = simulation.get_obstacle_lanelet(params.obstacle)
        if lanelet is not None:
            has_adj_lane_left = lanelet.adj_left_same_direction
            has_adj_lane_right = lanelet.adj_right_same_direction
        else:
            # Since lanelet is unspecified, we assume False to be an appropriate value
            has_adj_lane_left = False
            has_adj_lane_right = False

        return {
            V_Feature.HasAdjLaneLeft.value: 1.0 if has_adj_lane_left else 0.0,
            V_Feature.HasAdjLaneRight.value: 1.0 if has_adj_lane_right else 0.0,
        }
