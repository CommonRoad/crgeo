from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import Set2Set

from commonroad_geometric.learning.geometric.components.decoders.base_decoder import BaseDecoder


class Set2SetDecoder(BaseDecoder):
    def __init__(
        self,
        input_size: int,
        n: int,
        processing_steps: int = 5
    ):
        self._input_size = input_size
        self._n = n
        super().__init__()
        self.upscaler = nn.Linear(
            input_size,
            input_size*n
        )
        self.decoder = Set2Set(
            in_channels=input_size*n,
            processing_steps=processing_steps,
            num_layers=1
        )

    @property
    def output_size(self) -> int:
        return self._input_size

    def forward(
        self,
        n: int,
        x: Tensor,
        *state: Tensor
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError()

    def forward_n(self, x: Tensor, n: int) -> Tensor:
        batch_size = x.shape[0] if x.ndim > 1 else 1

        upscaled = self.upscaler(x)

        Y: Tensor = self.decoder(upscaled, batch=torch.arange(batch_size, device=x.device)).view(
            batch_size, self._n, 2*self.output_size
        )
        Y = Y[:, :, self.output_size:]
        if n > self._n:
            Y = torch.cat([
                Y,
                Y.new_zeros((batch_size, n - self._n, self.output_size))
            ], dim=1)
        else:
            Y = Y[:, :n, :]

        

        if x.ndim == 1:
            Y = Y.squeeze(0)

        return Y
