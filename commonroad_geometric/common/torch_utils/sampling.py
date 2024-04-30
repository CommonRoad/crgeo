from typing import Union
from torch import Tensor
import torch


def sample_indices(weights: Tensor, num_samples: Union[float, int] = 1.0) -> Tensor:
    if isinstance(num_samples, float):
        num_samples = int(weights.size(0) * num_samples)
    return torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=True,
    )
