from torch import nn
from torch.nn import BatchNorm1d

from commonroad_geometric.learning.geometric.components.decoders.base_decoder import BaseDecoder
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService


class MLPDecoder(BaseDecoder):
    def __init__(self, input_size, hidden_size, output_size, num_nodes: int, trial=None, optimizer_service: BaseOptimizerService = None, **kwargs) -> None:
        super().__init__()
        p = 0.1
        mlp_layers = 4
        hidden_channels = 16
        optimal_mlp_channels = [40, 49, 44, 22]
        optimal_mlp_dropouts = [0.32554234458841225, 0.40597304621115604, 0.13473646073784587, 0.12167903755185709]
        if optimizer_service is not None:
            mlp_layers =  optimizer_service.suggest_param("suggest_int", "n_decoder_mlp_layers", trial, **{'low': mlp_layers, 'high': 20})
        layers = []
        for i in range(mlp_layers):
            if optimizer_service is None and optimal_mlp_channels is not None:
                hidden_channels = optimal_mlp_channels[i]
                p = optimal_mlp_dropouts[i]
            else:
                hidden_channels = int(hidden_channels/2) if i > (mlp_layers-1)/2 else int(hidden_channels * 2)
            if optimizer_service is not None:
                hidden_channels =  optimizer_service.suggest_param("suggest_int", "decoder_mlp_channels{}".format(i), trial, **{'low': 2, 'high': 256})
                p = optimizer_service.suggest_param("suggest_float", "decoder_mlp_dropout_{}".format(i), trial, **{'low': 0.1, 'high': 0.5})
            layers.append(nn.Linear(input_size, hidden_channels))
            layers.append(BatchNorm1d(num_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            input_size = hidden_channels
        layers.append(nn.Linear(hidden_channels, output_size))
        layers.append(nn.Dropout(p))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x).relu()
        return out, None