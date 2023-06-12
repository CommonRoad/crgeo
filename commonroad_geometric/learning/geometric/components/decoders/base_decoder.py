from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor, nn


class BaseDecoder(ABC, nn.Module):
    @property
    @abstractmethod
    def output_size(self) -> int:
        ...

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        *state: Tensor
    ) -> Tuple[Tensor, ...]:
        ...

    def forward_n(self, x: Tensor, n: int) -> Tensor:
        batch_size = x.shape[0] if x.ndim > 1 else 1

        Y = x.new_zeros((
            batch_size,
            n,
            self.output_size
        ))
        state: Tuple[Tensor, ...] = tuple()
        for i in range(n):
            res = self(x, *state)
            if isinstance(res, Tensor):
                y = res
            else:
                y, *state = res
            Y[:, i] = y

        if x.ndim == 1:
            Y = Y.squeeze(0)

        return Y
