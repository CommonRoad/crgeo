from pathlib import Path

import torch
from commonroad_geometric.learning.geometric.base_geometric import STATE_DICT_FILE

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, InitializeTrainingCallbacksParams


class WarmStartCallback(BaseCallback[InitializeTrainingCallbacksParams]):
    def __init__(self, directory: Path):
        self._directory = directory

    def __call__(self, params: InitializeTrainingCallbacksParams) -> None:
        import os
        if os.path.exists(f'{self._directory}/{STATE_DICT_FILE}'):
            model_state = torch.load(f'{self._directory}/{STATE_DICT_FILE}', map_location=params.ctx.device)
            params.ctx.model.load_state_dict(model_state)
            params.ctx.optimizer.load_state_dict(torch.load(f'{self._directory}/optimizer.pt'))
