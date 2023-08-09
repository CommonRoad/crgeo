import torch
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.glob.glob import global_add_pool

from commonroad_geometric.learning.geometric.components.encoders.base_encoder import BaseEncoder


class GCNEncoder(torch.nn.Module, BaseEncoder):

    def __init__(self):
        super(GCNEncoder, self).__init__()

    def _build(self, num_features, hidden_channels, **kwargs) -> None:
        """Builds the architecture for the encoder

        Args:
            num_features (int): [number of input features],
            hidden_channels (int): [number of hidden channels],
        """
        self.conv = GCNConv(
            in_channels=num_features,
            out_channels=hidden_channels
        )

    def forward(self, x, edge_index, **kwargs):
        x = self.conv(x, edge_index).relu()
        # Aggregate the results
        return global_add_pool(x, kwargs['batch_indeces'])
