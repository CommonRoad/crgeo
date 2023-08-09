from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from commonroad_geometric.common.torch_utils.misc_transforms import signed_max
from commonroad_geometric.learning.geometric.components.decoders import BaseDecoder


class BaseRecurrentDecoder(BaseDecoder):
    def __init__(
        self,
        network_cls: nn.Module,
        network_options: Dict[str, Any],
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        in_bias: bool = False,
        out_bias: bool = True,
        include_h_output: bool = False,
        aggr: str = 'sum',
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.num_layers = num_layers
        self.aggr = aggr
        self._include_h_output = include_h_output

        self.network = network_cls(
            **network_options
        )

        if in_bias:
            self.in_bias = nn.Parameter(torch.zeros(input_size))
        else:
            self.in_bias = 0
        if out_bias:
            self.out_bias = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.out_bias = 0

    @property
    def output_size(self) -> int:
        return 2*self.hidden_size if self._include_h_output else self.hidden_size

    @abstractmethod
    def _init_recurrent_state(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        ...

    @abstractmethod
    def _next_recurrent_state(
        self,
        x: Tensor,
        q_last: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        ...

    def _compute_q_out(
        self,
        q: Tensor
    ) -> Tensor:
        if self.aggr == 'sum':
            return q.sum(dim=0)
        elif self.aggr == 'max':
            return torch.amax(q, dim=0)
        elif self.aggr == 'signed_max':
            return signed_max(q)
        else:
            raise NotImplementedError(self.aggr)

    def _compute_y(
        self,
        x: Tensor,
        q_out: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tensor:

        y_q = q_out + self.out_bias
        h = h_last[0].max(dim=0)[0]
        #x_h = x*h
        if self._include_h_output:
            y = torch.cat([y_q, h], dim=-1)
        else:
            y = y_q
        return y

    def forward(
        self,
        x: Tensor,
        q_last: Tensor = None,
        h_last: Tuple[Tensor, ...] = None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        assert not (q_last is None and h_last is not None)
        assert not (q_last is not None and h_last is None)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x_ = x[None, :, :].repeat(self.num_layers, 1, 1)

        if h_last is None:
            q_last, h_last = self._init_recurrent_state(x_)

        q, q_next, h_next = self._next_recurrent_state(
            x=x_,
            q_last=q_last,
            h_last=h_last
        )
        q_out = self._compute_q_out(q)
        y = self._compute_y(
            x=x,
            h_last=h_last,
            q_out=q_out
        )
        return y, q_next, h_next
