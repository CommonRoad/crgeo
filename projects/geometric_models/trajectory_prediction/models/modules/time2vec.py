import torch
from torch import nn, Tensor
import numpy as np


class Time2Vec(nn.Module):
    """Implementation of time representation from Time2Vec paper.
    https://arxiv.org/pdf/1907.05321.pdf

    Uses the sine function as periodic activation function.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim >= 1
        self.dim = dim
        self.frequency = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.phase_shift = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # initialize to values used in positional encoding from "Attention Is All You Need" paper
        freq = 1 / (10_000 ** (torch.arange(0, self.dim, step=2) / self.dim))
        self.frequency.data[0::2] = freq
        self.frequency.data[1::2] = freq[:int(self.dim / 2)]

        # alternate between sin and cos by alternating phase-shift between 0 and π/2
        self.phase_shift.data[0::2] = 0.0
        self.phase_shift.data[1::2] = np.pi / 2

    def forward(self, time: Tensor) -> Tensor:
        time = time[:, (0, )]  # TODO: hack
        assert time.size(-1) == 1
        time_vec = torch.empty(time.size()[:-1] + (self.dim,), dtype=time.dtype, device=time.device)
        time_vec[..., 0:1] = self.frequency[0] * time + self.phase_shift[0]
        time_vec[..., 1:] = torch.sin(self.frequency[1:] * time + self.phase_shift[1:])
        return time_vec
