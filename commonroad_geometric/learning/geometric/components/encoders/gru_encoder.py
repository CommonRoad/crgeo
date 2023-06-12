import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch_geometric.nn.models import MLP

from commonroad_geometric.learning.geometric.components.encoders.base_encoder import BaseEncoder
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService

MAX_LOGSTD = 10


class GRUEncoder(nn.Module, BaseEncoder):
    def __init__(self):
        super(GRUEncoder, self).__init__()

    def _build(self, num_features, trial=None, optimizer_service: BaseOptimizerService = None, **kwargs) -> None:
        super().__init__()
        p = 0.1
        num_layers = 5
        hidden_channels = 16
        mlp_channels = [16,32,16,8,1]
        if optimizer_service is not None:
            num_layers = optimizer_service.suggest_param("suggest_int", "n_encoder_layers", trial, **{'low': num_layers, 'high': 10})
            hidden_channels = optimizer_service.suggest_param("suggest_int", "n_encoder_layers", trial, **{'low': num_features, 'high': 10})
            p = optimizer_service.suggest_param("suggest_float", "encoder_dropout", trial, **{'low': p, 'high': 0.5})
            n_mlp_layers =  optimizer_service.suggest_param("suggest_int", "n_encoder_mlp_layers", trial, **{'low': 5, 'high': 20})
            mlp_channels = [hidden_channels]
            for i in range(n_mlp_layers):
                channel = optimizer_service.suggest_param("suggest_int", "n_encoder_gru_mlp_hidden_channel{}".format(i), trial, **{'low': num_features, 'high': 128})
                mlp_channels.append(channel)
        mlp_channels.append(1)
        self.gru = nn.GRU(num_features, hidden_channels, num_layers=num_layers, dropout=p, batch_first=True)
        self.linear_mu = MLP(channel_list=mlp_channels, dropout=p)
        self.linear_logstd = MLP(channel_list=mlp_channels, dropout=p)

    def forward(self, x, batch_indeces, training=True, *args, **kwargs):
        out, hidden = self.gru(x)
        if type(out) == PackedSequence:
            out = pad_packed_sequence(out, batch_first=True)[0]
        mu = self.linear_mu(out).squeeze(-1)
        __logstd__ = self.linear_logstd(out).clamp(max=MAX_LOGSTD).squeeze(-1)
        z = self.reparametrize(mu, __logstd__,training)
        return (z, out)

    def reparametrize(self, mu, logstd, training):
        if training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
