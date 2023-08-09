import math
from typing import Dict

from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import V2VFeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature
from commonroad_geometric.simulation.base_simulation import BaseSimulation

EPS = 1.0


class ClosenessFeatureComputer(BaseFeatureComputer[V2VFeatureParams]):

    def __init__(self, max_distance: float = 50.0) -> None:
        self._max_distance = max_distance
        self._max_log_distance = math.log(max_distance + EPS)
        super().__init__()

    def __call__(
        self,
        params: V2VFeatureParams,
        simulation: BaseSimulation,
    ) -> Dict[str, float]:
        distance = params.distance
        closeness = 1 - min(distance, self._max_distance) / self._max_distance
        log_closeness = 1 - min(math.log(distance + EPS), self._max_log_distance) / self._max_log_distance
        features = {
            V2V_Feature.Closeness.value: closeness,
            V2V_Feature.LogCloseness.value: log_closeness
        }

        return features
