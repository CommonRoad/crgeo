from typing import Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.activation import ReLU
from torch.nn.parameter import Parameter

from commonroad_geometric.learning.geometric.components.decoders.base_recurrent_decoder import BaseRecurrentDecoder
from commonroad_geometric.learning.geometric.components.mlp import MLP
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService


class BucketDecoder(BaseRecurrentDecoder):
    def __init__(
        self,
        input_size: int,
        net_arch: Union[int, None, Sequence[int]],
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        in_bias: bool = False,
        out_bias: bool = True,
        relu_q: bool = False,
        no_grad_q: bool = False,
        trainable_aggregation: bool = True,
        norm_method: str = 'none',
        activation_cls: Type[nn.Module] = ReLU,
        dropout: float = 0.0,
        optimizer_service: Optional[BaseOptimizerService] = None,
        **kwargs
    ):
        network_options = {}
        #network_options.update(kwargs)
        if hidden_size is None:
            hidden_size = input_size
        network_options['input_size'] = input_size + hidden_size
        network_options['output_size'] = hidden_size
        network_options['bias_in'] = in_bias
        network_options['net_arch'] = net_arch
        network_options['dropout'] = dropout
        network_options['norm_method'] = norm_method
        network_options['activation_cls'] = activation_cls

        self._no_grad_q = no_grad_q
        self._relu_q = relu_q

        super().__init__(
            network_cls=MLP,
            network_options=network_options,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_bias=in_bias,
            out_bias=out_bias,
            **kwargs
        )

        if trainable_aggregation:
            self.register_parameter(
                'alpha', Parameter(torch.zeros((1, )))
            )
        else:
            self.register_parameter(
                'alpha', Parameter(torch.ones((1, )), requires_grad=False)
            )

    def _init_recurrent_state(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        batch_size = x.shape[1]
        h_0 = (torch.cat([
            x.new_ones((self.num_layers, batch_size, self.hidden_size)
        )], dim=2),)
        q_0 = x.new_zeros((self.num_layers, batch_size, self.hidden_size))
        return (q_0, h_0)

    def _next_recurrent_state(
        self,
        x: Tensor,
        q_last: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        q = self._compute_q(
                x=x,
                q_last=q_last,
                h_last=h_last
            )
        q_next = self._compute_q_next(
            x=x,
            q_last=q_last,
            q=q
        )
        h_next = self._compute_h_next(
            x=x,
            q_last=q_last,
            q_next=q_next,
            h_last=h_last
        )
        return q, q_next, h_next

    def _compute_q(
        self,
        x: Tensor,
        q_last: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tensor:
        batch_size = x.shape[1]
        h_last_ = h_last[0]
        q_in = torch.cat([x, h_last_], dim=2)
        q = self.network(q_in.flatten(0, 1)).view(
            (self.num_layers, batch_size, self.hidden_size)
        )
        if self._relu_q:
            return torch.relu(q)
        else:
            return q

    def _compute_q_next(
        self,
        x: Tensor,
        q_last: Tensor,
        q: Tuple[Tensor, ...]
    ) -> Tensor:
        q_diff = torch.exp(self.alpha)*torch.relu(q)
        q_next_lin = q_last + q_diff
        if self._no_grad_q:
            with torch.no_grad():
                q_next = torch.clamp(q_next_lin, min=0.0, max=1e2)
        else:
            q_next = torch.clamp(q_next_lin, min=0.0, max=1e2)
        return q_next

    def _compute_h_next(
        self,
        x: Tensor,
        q_last: Tensor,
        q_next: Tensor,
        h_last: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:

        if self._no_grad_q:
            with torch.no_grad():
                h_next = (torch.exp(-q_next),)
        else:
            h_next = (torch.exp(-q_next),)
        return h_next
