import torch
from torch import nn

from commonroad_geometric.learning.geometric.components.encoders import BaseEncoder


class RNNEncoder(nn.Module, BaseEncoder):
    def __init__(self):
        super(RNNEncoder, self).__init__()

    def _build(self, num_features, hidden_channels, **kwargs) -> None:
        self.hidden_size = hidden_channels
        self.gru = nn.GRU(num_features, hidden_channels)
        return super()._build(num_features, hidden_channels, **kwargs)

    def forward(self, input, hidden, **kwargs):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self, num_nodes, device):
        return torch.zeros(1, num_nodes, self.hidden_size, device=device)