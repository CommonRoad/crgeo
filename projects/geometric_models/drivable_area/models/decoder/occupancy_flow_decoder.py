from typing import Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch_geometric.data import Batch, Data

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.common.torch_utils.sampling import sample_indices
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.geometric_models.drivable_area.models.decoder.common import UpsampleConv
from projects.geometric_models.drivable_area.models.decoder.transformer_decoder import TransformerDecoder


class OccupancyFlowDecoder(BaseModel):

    def __init__(
        self,
        cfg: Config,
        target_attribute: str,
        mask_attribute: str
    ):
        super().__init__()
        self.target_attribute = target_attribute
        self.mask_attribute = mask_attribute
        self.cfg = cfg

        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type == "ConvTranspose":
            # 2d transposed convolution decoder
            assert self.cfg.prediction_size == 64
            self.decoder = nn.Sequential(
                # 1 x 1 x node_feature_size
                nn.ConvTranspose2d(
                    in_channels=self.cfg.node_feature_size, out_channels=128,
                    kernel_size=4, stride=1, padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                # 4 x 4 x 128
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                # 8 x 8 x 64
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                # 16 x 16 x 32
                nn.ConvTranspose2d(
                    in_channels=32, out_channels=16,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                # 32 x 32 x 16
                nn.ConvTranspose2d(
                    in_channels=16, out_channels=2,
                    kernel_size=4, stride=2, padding=1,
                    bias=False,
                ),
                # 64 x 64 x 2
            )

        elif self.cfg.decoder_type == "Upsample Conv":
            # https://distill.pub/2016/deconv-checkerboard/
            self.decoder = nn.Sequential(
                # 1 x 1 x node_feature_size
                nn.Conv2d(
                    in_channels=self.cfg.node_feature_size, out_channels=256,
                    kernel_size=1, stride=1, padding=0,
                    bias=False,
                ),
                # 1 x 1 x 256
                UpsampleConv(out_size=4, in_channels=256, out_channels=128),
                # 4 x 4 x 128
                UpsampleConv(out_size=8, in_channels=128, out_channels=64),
                # 8 x 8 x 64
                UpsampleConv(out_size=16, in_channels=64, out_channels=32),
                # 16 x 16 x 32
                UpsampleConv(out_size=32, in_channels=32, out_channels=16),
                # 32 x 32 x 16
                UpsampleConv(out_size=64, in_channels=16, out_channels=2),
                # 64 x 64 x 2
            )

        elif self.cfg.decoder_type == "Transformer":
            self.decoder = TransformerDecoder(
                node_feature_size=self.cfg.node_feature_size,
                sigmoid_out=False
            )

        elif self.cfg.decoder_type == "MLP":
            # MLP decoder
            self.decoder = nn.Sequential(
                nn.Linear(self.cfg.node_feature_size, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 2 * self.cfg.prediction_size ** 2, bias=False),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(f"Unknown decoder_type config value: {self.cfg.decoder_type}")


    @overload
    def forward(self, data: CommonRoadData, x: Tensor, sampling_weights: Optional[Tensor] = None) -> Tensor:
        ...

    @overload
    def forward(self, data: CommonRoadData, x: Tensor,
                sampling_weights: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        ...

    def forward(self, data: CommonRoadData, x: Union[Tensor, Tuple[Union[Data, Batch], Tensor]],
                sampling_weights: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        N = x.size(0)
        assert_size(x, (N, self.cfg.node_feature_size))

        # only compute a prediction for a subset of samples during training
        sample_ind = None
        if sampling_weights is not None:
            try:
                sample_ind = sample_indices(sampling_weights, num_samples=self.cfg.training_resample_ratio)
            except RuntimeError:
                pass
            else:
                x = x[sample_ind]
                N = sample_ind.size(0)
                assert_size(x, (N, self.cfg.node_feature_size))

        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type in {"ConvTranspose", "Upsample Conv"}:
            x = x.view(N, self.cfg.node_feature_size, 1, 1)

        prediction = self.decoder(x)

        if self.cfg.decoder_type == "MLP":
            assert_size(prediction, (N, 2 * self.cfg.prediction_size ** 2))
        else:
            assert_size(prediction, (N, 2, self.cfg.prediction_size, self.cfg.prediction_size))
            prediction = prediction.view(N, -1)

        return prediction

    def compute_loss(
        self, 
        data: Union[CommonRoadData, CommonRoadDataTemporal], 
        prediction: Tensor, 
        only_ego: bool = False
    ) -> Tensor:
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
            target = data.v[self.target_attribute].index_select(
                0, sample_ind) if sample_ind is not None else data.v[self.target_attribute]
            mask = data.v[self.mask_attribute].index_select(
                0, sample_ind) if sample_ind is not None else data.v[self.mask_attribute]
        else:
            target = data.v[self.target_attribute]
            mask = data.v[self.mask_attribute]

        if only_ego:
            is_ego_mask = data.v.is_ego_mask.bool().squeeze(-1)
            target = target[is_ego_mask, ...]

        N = prediction.shape[0]
        prediction_size = self.cfg.prediction_size
        target = target.view(N, prediction_size, prediction_size, 2)
        mask = mask.view(N, prediction_size, prediction_size).bool()
        assert_size(prediction, (N, 2 * prediction_size ** 2))
        prediction = prediction.view(N, prediction_size, prediction_size, 2)
        
        loss = F.mse_loss(input=prediction[mask, ...].flatten(), target=target[mask, ...].flatten(), reduction="mean")
        return loss


