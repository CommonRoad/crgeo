import enum
from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.inits as pyg_inits
from torch import Tensor, nn
from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops, contains_self_loops, softmax


@enum.unique
class AttentionHeadReduction(enum.Enum):
    Concat = "concat"
    Average = "average"
    Maximum = "maximum"


class RoadCoverageEGATConv(pyg_nn.MessagePassing):

    # https://www.notion.so/GNN-592b7730b94f47ed94b29b3283b716ae

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        negative_slope: float = 0.2,
        att_heads: int = 1,
        att_heads_aggr: AttentionHeadReduction = AttentionHeadReduction.Concat,
        dropout_att_weights: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=-2)  # TODO node_dim=0 ?
        self.in_channels = in_channels
        self.out_channels: out_channels
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.att_heads = att_heads
        self.att_heads_aggr = att_heads_aggr
        assert 0.0 <= dropout_att_weights <= 1.0
        self.dropout_att_weights = dropout_att_weights

        # target_dist = central node
        # source = neighbor node
        self.lin_target = pyg_nn.Linear(
            in_channels, out_channels,
            bias=True,  # TODO bias?
            weight_initializer="glorot",
        )
        self.lin_source = pyg_nn.Linear(
            in_channels, out_channels,
            bias=True,  # TODO bias?
            weight_initializer="glorot",
        )
        self.lin_edge = pyg_nn.Linear(
            edge_dim, out_channels,
            bias=True,  # TODO bias?
            weight_initializer="glorot",
        )
        self.attention_weights = nn.Parameter(torch.Tensor(1, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))  # TODO bias?

        # TODO attentions heads

        self.reset_parameters()  # TODO

    def reset_parameters(self) -> None:
        self.lin_target.reset_parameters()
        self.lin_source.reset_parameters()
        pyg_inits.glorot(self.attention_weights)
        pyg_inits.zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        x_target = self.lin_target(x)
        x_source = self.lin_source(x)

        # add self loops
        fill_value = 0
        num_nodes = x_target.size(0)
        assert not contains_self_loops(edge_index)
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=fill_value, num_nodes=num_nodes)

        # edge_attr = self.lin_edge(edge_attr)  TODO
        out = self.propagate(edge_index, x=(x_target, x_source), edge_attr=edge_attr)
        out += self.bias
        return out

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        edge_attr = self.lin_edge(edge_attr)
        x = x_i + x_j + edge_attr

        x = F.leaky_relu(x, inplace=True, negative_slope=self.negative_slope)
        alpha = (x * self.attention_weights).sum(dim=-1)
        alpha = softmax(src=alpha, index=index, ptr=ptr, num_nodes=size_i)

        # optional edge dropout
        alpha = F.dropout(alpha, p=self.dropout_att_weights, training=self.training)

        return x_j * alpha.unsqueeze(-1)
