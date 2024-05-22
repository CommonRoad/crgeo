import os

import torch
import logging
from pathlib import Path

from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILE, STATE_DICT_FILE
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, CheckpointCallbackParams

logger = logging.getLogger(__name__)

class SaveCheckpointCallback(BaseCallback[CheckpointCallbackParams]):

    def __init__(
        self,
        directory: Path
    ):
        self._directory = directory
        self._call_count = 0
        
    def __call__(self, params: CheckpointCallbackParams):
        curr_directory = self._directory.joinpath(f"{type(params.ctx.model).__name__}_{params.ctx.epoch}")
        curr_directory.mkdir(parents=True, exist_ok=True)

        model_path = curr_directory.joinpath(MODEL_FILE)
        checkpoint_path = curr_directory.joinpath(STATE_DICT_FILE)
        optimizer_path = curr_directory.joinpath('optimizer.pt')

        params.ctx.model.save_model(model_path)
        params.ctx.model.save_state(checkpoint_path)
        torch.save(params.ctx.optimizer.state_dict(), optimizer_path)
        # if self._call_count == 0:
        #     # just testing
        #     params.ctx.model.load(model_path)
        self._call_count += 1
        
        logger.info(f"Saved model checkpoint to {curr_directory}")
