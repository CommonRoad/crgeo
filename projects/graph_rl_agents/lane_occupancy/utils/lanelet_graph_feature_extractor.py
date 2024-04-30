import logging
from typing import Optional

import gymnasium
import torch
from torch import Tensor

from commonroad_geometric.common.torch_utils.pygeo import get_batch_sizes
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.base_geometric_feature_extractor import BaseGeometricFeatureExtractor
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.v2l_encoder import V2LEncoderConfig
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.v2l_global_encoder import V2LGlobalEncoder
from projects.geometric_models.lane_occupancy.utils.preprocessing import preprocess_conditioning

logger = logging.getLogger(__name__)


class LaneletGraphFeatureExtractor(BaseGeometricFeatureExtractor):
    """
    GNN feature extractor with V2L -> L2L -> L2Ego interactions.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        path_length: float,
        device: str,
        encoding_size: int = 32,
        encoder_config: Optional[V2LEncoderConfig] = None,
        **_
    ):
        self._path_length = path_length
        self._encoding_size = encoding_size
        self._encoder_config = encoder_config
        super().__init__(observation_space, device=device)

    def _build(self, observation_space: gymnasium.Space, device: str) -> None:
        self._encoder = V2LGlobalEncoder(
            output_size=self._encoding_size,
            config=self._encoder_config
        )
        self._encoder.build(data=None, enable_batch_norm=False)
        self._encoder.to(device)

    @property
    def output_dim(self) -> int:
        return self._encoding_size

    def _forward(self, data: CommonRoadData) -> Tensor:
        # hack for relabeling walk tensor
        data.walks = data.walks.long()
        batch_size = int(data.vehicle.batch.max().item()) + 1
        walk_masks = data.ego_trajectory_sequence_mask.bool()
        padding_size = data.ego_trajectory_sequence_mask.shape[1]
        # walk_batch_sizes = data.ego_trajectory_sequence_mask.sum(dim=1).long()
        walks_flattened = data.walks.flatten().long()  # [walk_masks.flatten()]
        walk_batch = torch.arange(batch_size, dtype=torch.long, device=data.device).repeat_interleave(padding_size)
        lanelet_batch_sizes = get_batch_sizes(data.lanelet.batch)
        lanelet_batch_offset = torch.cumsum(
            torch.cat([torch.zeros([1], device=data.device), lanelet_batch_sizes], dim=0), 0)
        walk_batch_offsets = torch.index_select(lanelet_batch_offset, 0, walk_batch)
        data.walks = (walks_flattened + walk_batch_offsets).view(batch_size, -1)
        # data.l.batch.index_select(0, data.walks.flatten())

        # print(data.l.batch.long().index_select(0, data.walks.flatten().long()))

        try:
            preprocess_conditioning(
                data=data,
                walks=data.walks,
                walk_start_length=data.walk_start_length.squeeze(1),
                walk_velocity=None,
                path_length=self._path_length,
                walk_masks=walk_masks,
                ignore_assertion_errors=True
            )
            z, _, _ = self._encoder(data)
        except Exception as e:
            logger.error(e, exc_info=True)
            z = torch.zeros((batch_size, self._encoding_size), device=data.device, dtype=torch.float32)

        assert z.shape[0] == batch_size
        return z
