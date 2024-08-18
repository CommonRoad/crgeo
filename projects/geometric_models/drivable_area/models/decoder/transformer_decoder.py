import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch_geometric.data import Batch, Data

class TransformerDecoder(nn.Module):
    def __init__(self, node_feature_size, img_size=64, nhead=8, num_transformer_blocks=6, transformer_fc_dim=8, num_channels=1, sigmoid_out: bool=True):
        super().__init__()
        self.node_feature_size = node_feature_size
        self.img_size = img_size
        self.positional_embedding = nn.Parameter(torch.randn(1, img_size * img_size, transformer_fc_dim))
        self.linear = nn.Linear(node_feature_size, transformer_fc_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=transformer_fc_dim, nhead=nhead),
            num_layers=num_transformer_blocks
        )
        self.fc_out = nn.Linear(transformer_fc_dim, num_channels)
        self.sigmoid_out = sigmoid_out
        
    def forward(self, x):
        # x: [batch_size, node_feature_size]
        x = self.linear(x)  # [batch_size, transformer_fc_dim]
        x = x.unsqueeze(1).repeat(1, self.img_size * self.img_size, 1)  # repeat along sequence
        x += self.positional_embedding
        x = self.transformer(x, x)  # self-attention, could replace second x with a memory if available
        x = self.fc_out(x)
        x = x.permute(0, 2, 1).view(-1, 1, self.img_size, self.img_size)  # [batch_size, num_channels, img_size, img_size]
        return torch.sigmoid(x) if self.sigmoid_out else x
