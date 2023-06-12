from typing import Tuple

import torch
from torch import Tensor, nn

from commonroad_geometric.learning.geometric.components.decoders.base_decoder import BaseDecoder


class AttentionDecoder(BaseDecoder):
    def __init__(
        self,
        input_size: int
    ):
        super().__init__()
        self._input_size = input_size

        self.lin_query = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Tanh()
        )

    @property
    def output_size(self) -> int:
        return self._input_size

    def forward(
        self,
        x: Tensor,
        n: int,
        remainder: Tensor = None
    ) -> Tuple[Tensor, ...]:
        if remainder is None:
            remainder = x

        query = self.lin_query(remainder)
        unscaled_extraction = (query + 1e-4)*remainder
        
        extraction = torch.nn.functional.normalize(unscaled_extraction, p=1, dim=1)
        truncated_extraction = torch.minimum(extraction, remainder)

        next_remainder = remainder - truncated_extraction

        assert torch.all(next_remainder >= 0.0)

        scaled_truncated_extraction = truncated_extraction * x.shape[-1] / n

        return scaled_truncated_extraction, next_remainder
