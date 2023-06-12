import torch
from torch import nn
from torch_geometric.nn.glob.glob import global_add_pool
from torch_geometric.nn.models import MLP

from commonroad_geometric.learning.geometric.components.encoders.base_encoder import BaseEncoder
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService

MAX_LOGSTD = 10


class MLPEncoder(torch.nn.Module, BaseEncoder):
    def __init__(self):
        super(MLPEncoder, self).__init__()


    """MLP Encoder

    Args:
        num_features (int): [number of input features],
        hidden_channels (int): [number of hidden channels],
    """

    def _build(self, num_features, hidden_channels, trial = None, optimizer_service: BaseOptimizerService = None, **kwargs) -> None:
        num_layers = 2
        out_channels = hidden_channels
        mlp_channels = [num_features]
        num_layers = 7
        dropout_p = 0.20971573800837953
        optimal_channels = [18, 38, 121, 18, 75, 32, 56]
        if optimizer_service is not None:
            num_layers =  optimizer_service.suggest_param("suggest_int", "n_encoder_layers", trial, **{'low': num_layers, 'high': 20})
            dropout_p = optimizer_service.suggest_param("suggest_float", "encoder_dropout", trial, **{'low': 0.1, 'high': 0.5})
            for i in range(num_layers):
                channel = optimizer_service.suggest_param("suggest_int", "n_encoder_mlp_hidden_channel{}".format(i), trial, **{'low': 4, 'high': 256})
                mlp_channels.append(channel)
        else:
            mlp_channels = [num_features]
            for i in range(num_layers):
                mlp_channels.append(optimal_channels[i])
        
        mlp_out_channels = mlp_channels[-1]
        self.mlp = MLP(channel_list=mlp_channels, dropout=dropout_p)
        self.linear_mu = nn.Linear(mlp_channels[-1], out_channels)
        self.linear_logstd = nn.Linear(mlp_channels[-1], out_channels)

    def forward(self, x, batch_indeces, training=True, *args, **kwargs):
        x = self.mlp(x)
        x = global_add_pool(x, batch_indeces)
        mu = self.linear_mu(x)
        __logstd__ = self.linear_logstd(x).clamp(max=MAX_LOGSTD)
        z = self.reparametrize(mu, __logstd__,training)
        return (z, x)

    def reparametrize(self, mu, logstd, training):
        if training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
