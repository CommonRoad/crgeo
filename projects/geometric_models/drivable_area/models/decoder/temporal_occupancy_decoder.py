from typing import Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch_geometric.data import Batch, Data
import numpy as np

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.common.torch_utils.sampling import sample_indices
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.geometric_models.drivable_area.utils.confusion_matrix import BaseConfusionMatrix
from projects.geometric_models.drivable_area.models.decoder.common import UpsampleConv
from projects.geometric_models.drivable_area.models.decoder.transformer_decoder import TransformerDecoder

class TemporalOccupancyDecoder(BaseModel):

    def __init__(
        self,
        cfg: Config,
        target_attribute: str = "occupancy"
    ):
        super().__init__()
        self.cfg = cfg
        self.num_frames = self.cfg.num_temporal_frames
        self.target_attribute = target_attribute
    
        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type == "ConvTranspose":
            # 2d transposed convolution decoder
            assert self.cfg.prediction_size == 64
            self.decoder = nn.Sequential(
                # 1 x 1 x node_feature_size to num_frames x 4 x 4 x 128
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
                # num_frames x 32 x 32 x 16 to num_frames x 64 x 64 x 1
                nn.ConvTranspose3d(
                    in_channels=16, out_channels=1,
                    kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                    bias=False,
                ),
                nn.Sigmoid()
            )

        elif self.cfg.decoder_type == "Upsample Conv":
            self.decoder = nn.Sequential(
                # Input: 1 x 1 x node_feature_size
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
                UpsampleConv(out_size=64, in_channels=16, out_channels=1),
                # num_frames x 64 x 64 x 1
                nn.Sigmoid(),
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

        # == Road coverage prediction (decoder) ==
        if self.cfg.decoder_type in {"ConvTranspose", "Upsample Conv"}:
            x = x.view(N, self.cfg.node_feature_size, 1, 1, 1)

        prediction = self.decoder(x)
        # prediction in [0, 1]

        if self.cfg.decoder_type == "MLP":
            assert_size(prediction, (N, self.cfg.prediction_size ** 2))
        else:
            assert_size(prediction, (N, 1, self.cfg.num_temporal_frames, self.cfg.prediction_size, self.cfg.prediction_size))
            prediction = prediction.view(N, self.cfg.num_temporal_frames, self.cfg.prediction_size, self.cfg.prediction_size)

        return prediction

    def compute_loss(self, target: Tensor, prediction: Tensor, only_ego: bool = False) -> Tensor:
        """
        Compute the binary cross-entropy loss between the predicted and target matrices.

        Args:
            target (Tensor): The ground truth tensor.
                Shape: (batch_size, num_frames, prediction_size, prediction_size)
            prediction (Tensor): The predicted tensor.
                Shape: (batch_size, num_frames, prediction_size, prediction_size)
            only_ego (bool, optional): If True, compute loss only for ego vehicle. Defaults to False.

        Returns:
            Tensor: The computed loss.

        Raises:
            ValueError: If the input tensors don't match the expected shapes.
        """
        batch_size, num_frames, prediction_size, _ = prediction.shape

        # Validate input shapes
        if target.shape != prediction.shape:
            raise ValueError(f"Target shape {target.shape} does not match prediction shape {prediction.shape}")
        
        if prediction.shape[2] != prediction.shape[3] or prediction.shape[2] != self.cfg.prediction_size:
            raise ValueError(f"Prediction size {prediction.shape[2]} does not match configured size {self.cfg.prediction_size}")
        
        if num_frames != self.cfg.num_temporal_frames:
            raise ValueError(f"Number of frames {num_frames} does not match configured frames {self.cfg.num_temporal_frames}")

        # Reshape tensors for loss computation
        target_flat = target.reshape(batch_size, -1)
        prediction_flat = prediction.reshape(batch_size, -1)

        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy(input=prediction_flat, target=target_flat, reduction="mean")

        return loss



class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, x):
        return self.attention(x, x, x)[0]

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)
    

class TemporalOccupancyDecoderV2(BaseModel):
    def __init__(self, cfg: Config, target_attribute: str = "occupancy"):
        super().__init__()
        self.cfg = cfg
        self.num_frames = self.cfg.num_temporal_frames
        self.target_attribute = target_attribute

        # LSTM for temporal encoding
        # Motivation: Capture temporal dependencies in the input sequence
        if cfg.use_lstm:
            self.temporal_encoder = nn.LSTM(
                self.cfg.node_feature_size, 
                cfg.lstm_hidden_size, 
                num_layers=cfg.lstm_num_layers, 
                batch_first=True
            )
        
        # Temporal attention mechanism
        # Motivation: Allow the model to focus on the most relevant parts of the input sequence
        if cfg.use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                cfg.lstm_hidden_size if cfg.use_lstm else self.cfg.node_feature_size,
                cfg.temporal_attention_heads
            )
        
        decoder_input_size = (cfg.lstm_hidden_size if cfg.use_lstm else self.cfg.node_feature_size) * self.num_frames
        
        # Choose decoder architecture based on configuration
        if cfg.decoder_type == "ConvTranspose":
            self.decoder = self._build_conv_transpose_decoder(decoder_input_size)
        elif cfg.decoder_type == "Upsample":
            self.decoder = self._build_upsample_decoder(decoder_input_size)
        else:
            raise ValueError(f"Unknown decoder_type config value: {self.cfg.decoder_type}")

    def _build_conv_transpose_decoder(self, input_size):
        layers = []
        current_size = 1
        current_channels = input_size

        # Calculate the number of upsampling steps needed
        num_upsample_steps = int(np.log2(self.cfg.prediction_size))

        for i in range(num_upsample_steps):
            out_channels = self.cfg.conv_channels[min(i, len(self.cfg.conv_channels) - 1)]
            layers.extend([
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            if self.cfg.use_residual_blocks:
                layers.append(ResidualBlock(out_channels))
            current_channels = out_channels
            current_size *= 2

        layers.append(nn.Conv2d(current_channels, self.num_frames, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _build_upsample_decoder(self, input_size):
        layers = []
        current_size = 1
        current_channels = input_size

        # Calculate the number of upsampling steps needed
        num_upsample_steps = int(np.log2(self.cfg.prediction_size))

        for i in range(num_upsample_steps):
            out_channels = self.cfg.conv_channels[min(i, len(self.cfg.conv_channels) - 1)]
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            if self.cfg.use_residual_blocks:
                layers.append(ResidualBlock(out_channels))
            current_channels = out_channels
            current_size *= 2

        layers.append(nn.Conv2d(current_channels, self.num_frames, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, data: CommonRoadDataTemporal, x: Tensor, sampling_weights: Tensor = None) -> Tensor:
        N = x.size(0)
        assert_size(x, (N, self.cfg.node_feature_size))

        # Optional resampling for focused training
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

        # Prepare input for temporal processing
        x = x.unsqueeze(1).repeat(1, self.num_frames, 1)

        # Apply LSTM if configured
        if self.cfg.use_lstm:
            x, _ = self.temporal_encoder(x)

        # Apply temporal attention if configured
        if self.cfg.use_temporal_attention:
            x = self.temporal_attention(x)

        # Reshape for spatial decoding
        x = x.view(N, -1, 1, 1)
        prediction = self.decoder(x)

        assert_size(prediction, (N, self.num_frames, self.cfg.prediction_size, self.cfg.prediction_size))

        return prediction

    def compute_loss(self, target: Tensor, prediction: Tensor, only_ego: bool = False) -> Tensor:
        # Validate input shapes
        batch_size, num_frames, prediction_size, _ = prediction.shape

        if target.shape != prediction.shape:
            raise ValueError(f"Target shape {target.shape} does not match prediction shape {prediction.shape}")
        
        if prediction.shape[2] != prediction.shape[3] or prediction.shape[2] != self.cfg.prediction_size:
            raise ValueError(f"Prediction size {prediction.shape[2]} does not match configured size {self.cfg.prediction_size}")
        
        if num_frames != self.cfg.num_temporal_frames:
            raise ValueError(f"Number of frames {num_frames} does not match configured frames {self.cfg.num_temporal_frames}")

        # Flatten tensors for loss computation
        target_flat = target.reshape(batch_size, -1)
        prediction_flat = prediction.reshape(batch_size, -1)

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(input=prediction_flat, target=target_flat, reduction="mean")
        
        # Optionally combine with Dice loss for potentially better segmentation results
        if self.cfg.use_dice_loss:
            dice_loss = self.dice_loss(prediction_flat, target_flat)
            loss = self.cfg.bce_weight * bce_loss + (1 - self.cfg.bce_weight) * dice_loss
        else:
            loss = bce_loss

        return loss

    def dice_loss(self, pred, target, smooth=1.):
        # Compute Dice loss for better handling of class imbalance in segmentation
        pred = pred.contiguous()
        target = target.contiguous()    
        intersection = (pred * target).sum(dim=1)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)))
        return loss.mean()