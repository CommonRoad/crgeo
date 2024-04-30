from typing import Any, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import normal
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN


class L2LConvLayer(MessagePassing):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        edge_channels: int,
        mlp_hidden_channels: int,
        mlp_layers: int,
        aggr: str = 'max',
        act: Type[nn.Module] = nn.Identity,
        out_bias: bool = True,
        add_previous: bool = True,
        **mlp_kwargs: Any
    ):
        super(L2LConvLayer, self).__init__(aggr=aggr)

        self.bias = Parameter(torch.Tensor(output_size)) if out_bias else 0.0
        self.add_previous = add_previous
        self.msg_mlp = MLP(
            in_channels=2 * input_size + edge_channels,
            hidden_channels=mlp_hidden_channels,
            out_channels=output_size,
            num_layers=mlp_layers,
            **mlp_kwargs
        )

        self.act = act()

        self.reset_parameters()

    def reset_parameters(self):
        # normal(self.weights, mean=0, std=2)
        normal(self.bias, mean=0, std=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        msg_agg = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x
        )

        z = msg_agg + self.bias
        if self.add_previous:  # and z.shape[1] == x.shape[1]:
            y = x + z
        else:
            y = z
        if not self.training:
            self.message_intensities = msg_agg.abs().sum(dim=1).detach()
        return y

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        z = torch.column_stack([x_j, x_i, edge_attr])
        y_raw = self.msg_mlp.forward(z)
        y = self.act(y_raw)
        return y


class L2LGNN(BasicGNN):
    """
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """

    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return L2LConvLayer(in_channels, out_channels, **kwargs)


class V2LConvLayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        aggr: str,
        **mlp_kwargs: Any
    ):
        super(V2LConvLayer, self).__init__(aggr=aggr)

        self.msg_encoder = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            **mlp_kwargs
        )
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.bias, mean=0, std=2)
        self.msg_encoder.reset_parameters()

    def forward(self, x: Tuple[Tensor, Tensor], edge_index: Tensor, edge_attr: Tensor, dim_size: int) -> Tensor:
        msg_agg = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            dim_size=dim_size
        )

        z = msg_agg + self.bias
        return z

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        features = torch.column_stack([x_j, x_i, edge_attr])
        msg = self.msg_encoder.forward(features)
        return msg
