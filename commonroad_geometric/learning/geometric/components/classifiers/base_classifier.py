from abc import abstractmethod, ABC, abstractproperty
from typing import Tuple
from torch import nn, Tensor


class BaseClassfier(ABC, nn.Module):

    @abstractmethod
    def _build(self, input_size, hidden_channels, num_classes, **kwargs) -> None:
        """Base method for instantiation that every classifier must extend.
        """

    @abstractmethod
    def forward(
        self,
        input: Tensor,
        **kwargs
    ) -> Tuple[Tensor, ...]:
        ...
