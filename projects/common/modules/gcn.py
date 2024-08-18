from torch import nn, Tensor
import torch_geometric.nn as pyg_nn

import typing as t


class GCN(nn.Module):

    def __init__(
            self,
            layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: t.Optional[nn.Module] = None,
            p_dropout: float = 0.0,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.graph_convs = nn.ModuleList([
            pyg_nn.GCNConv(in_channels=input_dim if i == 0 else hidden_dim, out_channels=hidden_dim, add_self_loops=True)
            for i in range(layers)
        ])
        self.activation = activation
        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index=edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        return self.linear(x)
