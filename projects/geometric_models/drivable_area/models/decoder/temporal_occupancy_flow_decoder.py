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

class TemporalOccupancyFlowDecoder(BaseModel):

    def __init__(
        self,
        cfg: Config,
        target_attribute: str,
        mask_attribute: str
    ):
        super().__init__()
        self.cfg = cfg
        self.target_attribute = target_attribute
        self.mask_attribute = mask_attribute
        self.num_frames = self.cfg.num_temporal_frames
    
        # == Temporal Occupancy Flow prediction (decoder) ==
        if self.cfg.decoder_type == "ConvTranspose":
            # 3d transposed convolution decoder
            assert self.cfg.prediction_size == 64
            self.decoder = nn.Sequential(
                # 1 x 1 x 1 x node_feature_size to num_frames x 4 x 4 x 128
                nn.ConvTranspose3d(
                    in_channels=self.cfg.node_feature_size, 
                    out_channels=128,
                    kernel_size=(self.num_frames, 4, 4), stride=1, padding=0,
                    bias=False,
                ),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                # num_frames x 4 x 4 x 128 to num_frames x 8 x 8 x 64
                nn.ConvTranspose3d(
                    in_channels=128, out_channels=64,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                # num_frames x 8 x 8 x 64 to num_frames x 16 x 16 x 32
                nn.ConvTranspose3d(
                    in_channels=64, out_channels=32,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                # num_frames x 16 x 16 x 32 to num_frames x 32 x 32 x 16
                nn.ConvTranspose3d(
                    in_channels=32, out_channels=16,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                # num_frames x 32 x 32 x 16 to num_frames x 64 x 64 x 2
                nn.ConvTranspose3d(
                    in_channels=16, out_channels=2,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                    bias=False,
                ),
            )

        elif self.cfg.decoder_type == "Upsample Conv":
            self.decoder = nn.Sequential(
                # Input: 1 x 1 x 1 x node_feature_size
                nn.Conv3d(
                    in_channels=self.cfg.node_feature_size, out_channels=256,
                    kernel_size=(1, 1, 1), stride=1, padding=0,
                    bias=False,
                ),
                # Convert to 2D
                nn.Flatten(start_dim=2, end_dim=-1),
                nn.Unflatten(1, (self.num_frames, 256)),
                # num_frames x 1 x 1 x 256
                UpsampleConv(out_size=4, in_channels=256, out_channels=128),
                # num_frames x 4 x 4 x 128
                UpsampleConv(out_size=8, in_channels=128, out_channels=64),
                # num_frames x 8 x 8 x 64
                UpsampleConv(out_size=16, in_channels=64, out_channels=32),
                # num_frames x 16 x 16 x 32
                UpsampleConv(out_size=32, in_channels=32, out_channels=16),
                # num_frames x 32 x 32 x 16
                UpsampleConv(out_size=64, in_channels=16, out_channels=2),
                # num_frames x 64 x 64 x 2
            )

        else:
            raise ValueError(f"Unknown decoder_type config value: {self.cfg.decoder_type}")

    def forward(self, data: CommonRoadDataTemporal, x: Union[Tensor, Tuple[Union[Data, Batch], Tensor]],
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

        # == Temporal Occupancy Flow prediction (decoder) ==
        if self.cfg.decoder_type in {"ConvTranspose", "Upsample Conv"}:
            x = x.view(N, self.cfg.node_feature_size, 1, 1, 1)

        prediction = self.decoder(x)

        assert_size(prediction, (N, 2, self.num_frames, self.cfg.prediction_size, self.cfg.prediction_size))
        prediction = prediction.permute(0, 2, 3, 4, 1)  # N x num_frames x prediction_size x prediction_size x 2

        return prediction

    def compute_loss(self, target: Tensor, prediction: Tensor, only_ego: bool = False) -> Tensor:
        """
        Compute the mean squared error loss between the predicted and target flow matrices.

        Args:
            target (Tensor): The ground truth flow tensor.
                Shape: (batch_size, num_frames, prediction_size, prediction_size, 2)
            prediction (Tensor): The predicted flow tensor.
                Shape: (batch_size, num_frames, prediction_size, prediction_size, 2)
            only_ego (bool, optional): If True, compute loss only for ego vehicle. Defaults to False.
                This parameter is not used in the current implementation but kept for consistency.

        Returns:
            Tensor: The computed loss.

        Raises:
            ValueError: If the input tensors don't match the expected shapes.
        """
        batch_size, num_frames, prediction_size, _, num_channels = prediction.shape

        # Validate input shapes
        if target.shape != prediction.shape:
            raise ValueError(f"Target shape {target.shape} does not match prediction shape {prediction.shape}")
        
        if prediction.shape[2] != prediction.shape[3] or prediction.shape[2] != self.cfg.prediction_size:
            raise ValueError(f"Prediction size {prediction.shape[2]} does not match configured size {self.cfg.prediction_size}")
        
        if num_frames != self.cfg.num_temporal_frames:
            raise ValueError(f"Number of frames {num_frames} does not match configured frames {self.cfg.num_temporal_frames}")

        if num_channels != 2:
            raise ValueError(f"Number of channels {num_channels} does not match expected 2 (x and y flow)")

        # Compute mean squared error loss
        loss = F.mse_loss(input=prediction, target=target, reduction="mean")

        return loss