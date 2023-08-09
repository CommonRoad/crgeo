import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob.glob import global_max_pool
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.autoencoder import MAX_LOGSTD

from commonroad_geometric.learning.geometric.components.encoders.base_encoder import BaseEncoder
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService


class VGAEEncoder(torch.nn.Module, BaseEncoder):

    def __init__(self):
        super(VGAEEncoder, self).__init__()

    def _build(self, num_features, hidden_channels, trial=None, optimizer_service: BaseOptimizerService = None, **kwargs) -> None:
        """Variational Autoencoder

        Args:
            num_features (int): [number of input features],
            hidden_channels (int): [number of hidden channels],
        """
        layers = []
        out_channels = hidden_channels
        mlp_out_channels = 16
        num_layers = 7
        dropout_p = 0.2

        gcn_optimal_channels = [num_features, 16, 32, 64, 32]
        self._gcn_optimal_dropouts = [0.1, 0.1, 0.1, 0.1 ,0.1]

        n_gcn_layers = len(gcn_optimal_channels)
        self._gcns = torch.nn.ModuleList()
        self._norms = torch.nn.ModuleList()
        self._act = ReLU()
        if optimizer_service is not None:
            n_gcn_layers = trial.suggest_int("n_encoder_gcn_layers", 3, 10)
            gcn_optimal_channels = [num_features]
            gcn_optimal_dropouts = [0.1]
            for i in range(n_gcn_layers-1):
                channel = optimizer_service.suggest_param("suggest_int", "n_encoder_gcn_features{}".format(i), trial, **{'low': mlp_out_channels, 'high': 256})
                dropout_p = optimizer_service.suggest_param("suggest_float", "gcn_dropout", trial, **{'low': 0.1, 'high': 0.5})
                gcn_optimal_channels.append(channel)
                gcn_optimal_dropouts.append(dropout_p)

        for i in range(n_gcn_layers):
            self._gcns.append(GCNConv(num_features, gcn_optimal_channels[i]))
            self._norms.append(BatchNorm1d(gcn_optimal_channels[i]))
            num_features = gcn_optimal_channels[i]

        optimal_mlp_channels = [gcn_optimal_channels[-1], 18, 38, 121, 18, 75, 32, 56]

        if optimizer_service is not None:
            num_layers =  optimizer_service.suggest_param("suggest_int", "n_mlp_encoder_layers", trial, **{'low': num_layers, 'high': 10})
            dropout_p = optimizer_service.suggest_param("suggest_float", "mlp_dropout", trial, **{'low': 0.1, 'high': 0.5})
            optimal_mlp_channels = [gcn_optimal_channels[-1]]
            for i in range(num_layers-1):
                channel = optimizer_service.suggest_param("suggest_int", "n_encoder_mlp_hidden_channel{}".format(i), trial, **{'low': mlp_out_channels, 'high': 128})
                optimal_mlp_channels.append(channel)

        optimal_mlp_channels.append(out_channels)

        self.mu = MLP(channel_list=optimal_mlp_channels, dropout=dropout_p)
        self.logstd = MLP(channel_list=optimal_mlp_channels, dropout=dropout_p)

    def gcn_forward(self, x, edge_index):
        for i, (gcn, norm) in enumerate(zip(self._gcns, self._norms)):
            x = gcn(x, edge_index)
            x = norm(x)
            x = self._act(x)
            x = F.dropout(x, p=self._gcn_optimal_dropouts[i], training=self.training)
        return x


    def forward(self, x, edge_index, batch_indeces, training=True):
        x = self.gcn_forward(x, edge_index)
        z = global_max_pool(x, batch_indeces)

        mu = self.mu(z)
        logstd = self.logstd(z).clamp(max=MAX_LOGSTD)
        # Reparameterization trick to make z differentiable
        x = self.reparametrize(mu, logstd)
        # Aggregation for the results
        return (x, z)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
