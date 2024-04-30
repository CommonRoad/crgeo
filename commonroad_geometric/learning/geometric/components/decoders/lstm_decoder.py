from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.rnn import LSTM

from commonroad_geometric.learning.geometric.components.decoders.base_recurrent_decoder import BaseRecurrentDecoder


class LSTMDecoder(BaseRecurrentDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        in_bias: bool = False,
        out_bias: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        network_options = dict(
            input_size=input_size,
            num_layers=num_layers
        )
        if hidden_size is not None:
            network_options['proj_size'] = hidden_size
            network_options['hidden_size'] = input_size + hidden_size
            network_options['input_size'] = hidden_size
        else:
            network_options['proj_size'] = input_size
            network_options['hidden_size'] = 2 * input_size
            network_options['input_size'] = input_size
        network_options['dropout'] = dropout
        # network_options.update(kwargs)
        # del network_options['net_arch']
        super().__init__(
            network_cls=LSTM,
            network_options=network_options,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_bias=in_bias,
            out_bias=out_bias,
            **kwargs
        )

    def _init_recurrent_state(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        batch_size = x.shape[1]
        h_0 = (
            x.new_zeros((self.num_layers, batch_size, self.hidden_size)),
            torch.cat([
                x + self.in_bias, x.new_zeros((self.num_layers, batch_size, self.hidden_size)
                                              )], dim=2),
        )
        q_0 = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
        return (q_0, h_0)

    def _next_recurrent_state(
        self,
        x: Tensor,
        q_last: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        q, h_next = self.network.forward(q_last, h_last)
        q_next = q
        return q, q_next, h_next
