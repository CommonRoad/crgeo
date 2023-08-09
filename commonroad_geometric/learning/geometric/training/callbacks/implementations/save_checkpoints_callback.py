import os

import torch
import dill

from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILE, STATE_DICT_FILE
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, CheckpointCallbackParams


class SaveCheckpointCallback(BaseCallback[CheckpointCallbackParams]):

    def __init__(
        self,
        directory: str
    ):
        self._directory = directory
        self._checkpoint_path = f'{self._directory}/{STATE_DICT_FILE}'
        self._model_path = f'{self._directory}/{MODEL_FILE}'
        self._call_count = 0

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def model_path(self) -> str:
        return self._model_path

    def __call__(self, params: CheckpointCallbackParams):
        os.makedirs(self._directory, exist_ok=True)
        params.ctx.model.save_model(self._model_path)
        params.ctx.model.save_state(self._checkpoint_path)
        torch.save(params.ctx.optimizer.state_dict(), f'{self._directory}/optimizer.pt')
        if self._call_count == 0:
            # just testing
            params.ctx.model.load(self._model_path)
        self._call_count += 1
