from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch_geometric.nn import BatchNorm

from commonroad_geometric.learning.geometric.components.decoders import BaseDecoder
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService


class SequentialDecoder(BaseDecoder):

    def __init__(self, input_size, hidden_size, output_size, num_nodes: int, trial=None, optimizer_service: BaseOptimizerService = None, **kwargs) -> None:
        super().__init__()
        p = 0.1
        num_layers = 2
        mlp_layers = 4
        hidden_channels = 16
        hidden_size = 63
        optimal_mlp_channels = [40, 49, 44, 22]
        optimal_mlp_dropouts = [0.32554234458841225, 0.40597304621115604, 0.13473646073784587, 0.12167903755185709]
        if optimizer_service is not None:
            num_layers =  optimizer_service.suggest_param("suggest_int", "n_decoder_layers", trial, **{'low': num_layers, 'high': 20})
            hidden_size =  optimizer_service.suggest_param("suggest_int", "decoder_hidden_size", trial, **{'low': hidden_size, 'high': 256})
            mlp_layers =  optimizer_service.suggest_param("suggest_int", "n_decoder_mlp_layers", trial, **{'low': mlp_layers, 'high': 20})
        else:
            mlp_layers = len(optimal_mlp_channels)
        self.gru = nn.GRU(input_size, hidden_size, num_layers = num_layers, dropout = p, batch_first=True)

        layers = []
        input_size = hidden_size
        for i in range(mlp_layers):
            if optimizer_service is None and optimal_mlp_channels is not None:
                hidden_channels = optimal_mlp_channels[i]
                p = optimal_mlp_dropouts[i]
            else:
                hidden_channels = int(hidden_channels/2) if i > (mlp_layers-1)/2 else int(hidden_channels * 2)
            if optimizer_service is not None:
                hidden_channels =  optimizer_service.suggest_param("suggest_int", "decoder_mlp_channels{}".format(i), trial, **{'low': 16, 'high': 128})
                p = optimizer_service.suggest_param("suggest_float", "decoder_mlp_dropout_{}".format(i), trial, **{'low': 0.1, 'high': 0.5})
            layers.append(nn.Linear(input_size, hidden_channels))
            layers.append(nn.Dropout(p))
            layers.append(BatchNorm(num_nodes))
            layers.append(nn.ReLU())
            input_size = hidden_channels
        layers.append(nn.Linear(hidden_channels, output_size))
        layers.append(nn.Dropout(p))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out, hidden = self.gru(x)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        if type(out) == PackedSequence:
            out = pad_packed_sequence(out, batch_first=True)[0]
        out = self.mlp(out).relu()
        return out, hidden
