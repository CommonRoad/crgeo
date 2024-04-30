from dataclasses import dataclass
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from torch_geometric.nn.models import MLP
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import Tensor

from projects.geometric_models.lane_occupancy.models.occupancy.decoders.base_occupancy_decoder import BaseOccupancyDecoder, BaseOccupancyDecoderConfig


EPS = 1e6


@dataclass
class MLPOccupancyDecoderConfig(BaseOccupancyDecoderConfig):
    mlp_hidden_size: int = 256
    mlp_layers: int = 3


class MLPOccupancyDecoder(BaseOccupancyDecoder):

    def __init__(
        self,
        input_size: int,
        offset_conditioning: bool,
        config: Optional[MLPOccupancyDecoderConfig] = None,
        **kwargs: Any
    ):
        config = config or MLPOccupancyDecoderConfig(**kwargs)
        super(MLPOccupancyDecoder, self).__init__()

        self.input_size = input_size
        self.offset_conditioning = offset_conditioning
        self.config = config

    def reset_config(self) -> None:
        self.config = MLPOccupancyDecoderConfig()  # TODO

    def build(
        self,
        data: CommonRoadData,
        trial = None
    ) -> None:
        self.decoder = MLP(
            in_channels=self.input_size + 2,
            out_channels=1,
            hidden_channels=self.config.mlp_hidden_size,
            num_layers=self.config.mlp_layers
        )

    def reset_parameters(self) -> None:
        """Initializes model weights.
        """
        super(MLPOccupancyDecoder, self).reset_parameters()

    def forward(
        self,
        z: Tensor,
        lanelet_length: Union[float, Tensor],
        domain: Union[int, Tensor],
        dt: float,
        time_horizon: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size = z.shape[0]
        device = z.device

        if isinstance(domain, int):
            domain = torch.linspace(
                0, 1, domain, device=device
            )[None, None, None, :].repeat(
                1, batch_size, time_horizon, 1
            )
        else:
            if domain.ndim == 3:
                domain = domain.unsqueeze(0)
        domain = domain.unsqueeze(-1)

        resolution = domain.shape[3]

        t = dt * torch.arange(time_horizon, device=device
                              )[None, None, :, None, None].repeat(
            domain.shape[0], batch_size, 1, resolution, 1
        )

        z_view = z[None, :, None, None, :].repeat(domain.shape[0], 1, time_horizon, resolution, 1)

        del z

        features = torch.cat([
            domain, t, z_view
        ], dim=-1)

        del z_view

        occ_probs = torch.sigmoid(
            self.decoder(features.view(-1, self.input_size + 2)
                         ).view(domain.shape[0], batch_size, time_horizon, resolution)
        )

        del features

        info = dict(
            # features=features
        )

        occ_probs = occ_probs.squeeze(0)

        return occ_probs, info
