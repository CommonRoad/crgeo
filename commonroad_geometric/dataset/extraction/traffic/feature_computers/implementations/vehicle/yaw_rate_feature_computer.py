from collections import defaultdict
from typing import Dict

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class YawRateFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    def __init__(self) -> None:
        self._last_orientation_cache: Dict[int, float] = defaultdict(float)
        super().__init__()

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:

        obstacle_id = params.obstacle.obstacle_id
        orientation = params.state.orientation
        if obstacle_id in self._last_orientation_cache:
            last_orientation = self._last_orientation_cache.get(obstacle_id)
            orientation_difference = relative_orientation(last_orientation, orientation)
            yaw_rate = orientation_difference / params.dt
        else:
            yaw_rate = 0.0

        self._last_orientation_cache[obstacle_id] = orientation

        features = {
            V_Feature.YawRate.value: yaw_rate
        }

        return features

    def _reset(self, simulation: BaseSimulation) -> None:
        self._last_orientation_cache = defaultdict(float)
