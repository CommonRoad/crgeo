from typing import Set

import torch

from crgeo.common.class_extensions.class_property_decorator import classproperty
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from crgeo.simulation.base_simulation import BaseSimulation


class VehicleVerticesFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    @classproperty
    def skip_normalize_features(cls) -> Set[str]:
        return {V_Feature.Vertices.value}
        
    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        features = {
            V_Feature.Vertices.value: torch.from_numpy(params.obstacle.obstacle_shape.vertices).to(torch.float32).flatten()
        }

        return features
