from typing import Union, Callable
import torch


def sample_indices(weights: torch.Tensor, num_samples: Union[float, int] = 1.0) -> torch.Tensor:
    if isinstance(num_samples, float):
        num_samples = int(weights.size(0) * num_samples)
    return torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=True,
    )
