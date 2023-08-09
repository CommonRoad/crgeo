from commonroad_geometric.learning.geometric.components.classifiers.base_classifier import BaseClassfier
from torch import nn, Tensor
from typing import Tuple

from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService
from torch_geometric.nn import BatchNorm

class LinearClassifier(BaseClassfier):
    def __init__(self) -> None:
        super(BaseClassfier, self).__init__()

    def _build(self, input_size, hidden_channels: int, num_classes: int, dropout_p: float=0.5, trial = None, optimizer_service: BaseOptimizerService = None, add_batch_norm = True, **kwargs) -> None:
        layers = []
        n_layers = 5
        p = 0.1
        best_layer_params = [21, 70, 67, 13, 104, 12, 23, 18, 69, 70, 36, 119, 69, 109, 100, 6, 59, 22]
        best_dropout_params = [0.46009886685470147, 0.28880394576993484, 
        0.47756234743131964, 0.3721294775196487, 0.2874747504551072,
        0.36589905910540116, 0.4399668378475185, 0.3558543991706663,
        0.4115124856268484, 0.3330426081069601, 0.48854852727681236,
        0.45072804245495096, 0.3584999006601297, 0.31132534919545535,
        0.4636405983325905, 0.38103943275574237, 0.42998562138318536,
        0.304656012162299]

        if optimizer_service is not None:
            n_layers =  optimizer_service.suggest_param("suggest_int", "n_classifier_layers", trial, **{'low': 2, 'high': 20})
        else:
            n_layers = len(best_layer_params)
        for i in range(n_layers):
            if optimizer_service is not None:
                hidden_channels = optimizer_service.suggest_param("suggest_int", "n_classifier_out_features_l{}".format(i), trial, **{'low': 4, 'high': 128})
                p = optimizer_service.suggest_param("suggest_float", "classifier_dropout_{}".format(i), trial, **{'low': 0.2, 'high': 0.5})
            else:
                hidden_channels = best_layer_params[i]
                p = best_dropout_params[i]
            layers.append(nn.Linear(input_size, hidden_channels))
            layers.append(nn.Dropout(p))
            layers.append(nn.ReLU())
            if add_batch_norm:
                layers.append(BatchNorm(hidden_channels))
            input_size = hidden_channels
        layers.append(nn.Linear(hidden_channels, num_classes))
        self.lin = nn.Sequential(*layers)
    
    def forward(
        self,
        input,
        **kwargs
    ) -> Tuple[Tensor, ...]:
        return self.lin(input)
