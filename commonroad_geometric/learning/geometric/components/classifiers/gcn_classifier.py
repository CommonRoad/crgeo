import torch
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn.functional as F
from commonroad_geometric.learning.geometric.components.classifiers.base_classifier import BaseClassfier

class GCNClassifier(BaseClassfier):
    def __init__(self):
        super(GCNClassifier, self).__init__()
    
    def _build(self, num_features: int, hidden_channels: int, num_classes: int, dropout_p: float=0.5):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_p=dropout_p

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        return x
