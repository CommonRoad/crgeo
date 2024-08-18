from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity, Linear


class MLP(torch.nn.Module):
    r"""A Multi-Layer Perception (MLP) model.
    Original code from torch_geometric/nn/models/mlp.py

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        activation_fn (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        activation_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        batch_norm_kwargs (Dict[str, Any], optional): Arguments passed to
            :class:`torch.nn.BatchNorm1d` in case :obj:`batch_norm == True`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
    """
    def __init__(
        self,
        *,
        channel_list: list[int],
        dropout: float = 0.,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        activation_first: bool = False,
        batch_norm: bool = True,
        batch_norm_kwargs: Optional[dict[str, Any]] = None,
        bias: bool = True,
        residual_connections: bool = False,
    ):
        super().__init__()
        batch_norm_kwargs = batch_norm_kwargs or {}

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.residual_connections = residual_connections
        if residual_connections:
            if len(self.channel_list) == 2:
                assert self.channel_list[0] == self.channel_list[1]
            else:  # > 2
                assert all(d == self.channel_list[1] for d in self.channel_list[2:])

        self.dropout = dropout
        self.activation_fn = activation_fn
        self.activation_first = activation_first

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        if self.residual_connections and self.channel_list[0] == self.channel_list[1]:
            x = x + self.lins[0](x)
        else:
            x = self.lins[0](x)

        for lin, norm in zip(self.lins[1:], self.norms):
            x_update = x
            if self.activation_first:
                x_update = self.activation_fn(x_update)
            x_update = norm(x_update)
            if not self.activation_first:
                x_update = self.activation_fn(x_update)
            x_update = F.dropout(x_update, p=self.dropout, training=self.training)

            if self.residual_connections:
                x = x + lin.forward(x_update)
            else:
                x = lin.forward(x_update)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"
