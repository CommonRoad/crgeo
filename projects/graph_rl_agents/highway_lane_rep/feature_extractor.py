import gymnasium

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.base_geometric_feature_extractor import BaseGeometricFeatureExtractor
from torch_geometric.utils import to_dense_batch
from torch import Tensor
import torch


class HighwayLaneRepFeatureExtractor(BaseGeometricFeatureExtractor):

    def __init__(
        self, 
        observation_space: gymnasium.Space
    ):
        self._output_dim = 108 # todo
        super().__init__(observation_space)

    def _build(self, observation_space: gymnasium.Space) -> None:
        ...

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _forward(self, data: CommonRoadData) -> Tensor:
        concat_encodings = to_dense_batch(data.l.occupancy_encodings, data.l.batch)[0].flatten(start_dim=1)
        
        ego_obs = data.ego_observation
        road_obs = data.road_observation
        goal_obs = data.goal_observation

        x = torch.cat([ 
            ego_obs, 
            road_obs,
            goal_obs,
            concat_encodings,
        ], dim=-1)

        # output_dim: x.shape[-1]

        return x
