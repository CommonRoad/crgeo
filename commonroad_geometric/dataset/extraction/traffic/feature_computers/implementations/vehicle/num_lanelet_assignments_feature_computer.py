from typing import Dict, Set

import numpy as np
import torch
from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty

from commonroad_geometric.common.geometry.helpers import rotate_2d_matrix
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class NumLaneletAssignmentsFeatureComputer(BaseFeatureComputer[VFeatureParams]):

    @classproperty
    def skip_normalize_features(cls) -> Set[str]:
        return {V_Feature.NumLanaletAssignments.value}

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        features = {
            V_Feature.NumLanaletAssignments.value: len(simulation.obstacle_id_to_lanelet_id[params.obstacle.obstacle_id]),
        }

        return features
