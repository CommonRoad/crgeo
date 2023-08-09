import time
from pathlib import Path

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, CheckpointCallbackParams
from commonroad_geometric.learning.geometric.training.callbacks.implementations.save_checkpoints_callback import SaveCheckpointCallback
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features


class EpochCheckpointCallback(BaseCallback[CheckpointCallbackParams]):
    def __init__(
        self,
        directory: Path,
        checkpoint_cooldown: float = 0.0
    ):
        self._directory = directory
        self._last_checkpoint_time = time.time() - checkpoint_cooldown - 1
        self._checkpoint_cooldown = checkpoint_cooldown
        self._save_checkpoint = SaveCheckpointCallback(directory=self._directory)

    def __call__(self, params: CheckpointCallbackParams) -> None:
        current_time = time.time()
        time_since_last_checkpoint = current_time - self._last_checkpoint_time
        if time_since_last_checkpoint > self._checkpoint_cooldown:
            best_validation_loss = len(params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value]) > 0 and \
                                   params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value][-1] == min(
                params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value])
            if best_validation_loss:
                self._last_checkpoint_time = current_time
                self._save_checkpoint(params)
