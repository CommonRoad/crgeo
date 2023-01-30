from typing import Set, TYPE_CHECKING

import numpy as np
import torch

from crgeo.common.geometry.helpers import rotate_2d_matrix
from crgeo.common.class_extensions.class_property_decorator import classproperty
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from crgeo.dataset.extraction.traffic.feature_computers.implementations.types import V_Feature
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, VFeatureParams
from crgeo.simulation.base_simulation import BaseSimulation


class EgoFramePoseFeatureComputer(BaseFeatureComputer[VFeatureParams]):

    @classproperty
    def skip_normalize_features(cls) -> Set[str]:
        return {V_Feature.AngleEgoFrame.value}

    def __call__(
        self,
        params: VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        # TODO: Unit test
        
        pos = params.state.position
        v = params.state.velocity * np.array([np.cos(params.state.orientation), np.sin(params.state.orientation)])

        pos_rel = pos - params.ego_state.position
        rotation = rotate_2d_matrix(-params.ego_state.orientation)
        pos_ego_frame = rotation @ pos_rel
        v_ego_frame = rotation @ v
        angle_ego_frame = np.arctan2(pos_ego_frame[1], pos_ego_frame[0])

        features = {
            V_Feature.PosEgoFrame.value: torch.from_numpy(pos_ego_frame),
            V_Feature.AngleEgoFrame.value: angle_ego_frame,
            V_Feature.VelocityEgoFrame.value: torch.from_numpy(v_ego_frame)
        }

        return features
