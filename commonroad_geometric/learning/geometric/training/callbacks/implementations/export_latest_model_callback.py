from pathlib import Path
from typing import Optional
import logging
from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILE
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, CheckpointCallbackParams, StepCallbackParams
from commonroad_geometric.learning.geometric.training.callbacks.implementations.save_checkpoints_callback import SaveCheckpointCallback
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features


logger = logging.getLogger(__name__)


class ExportLatestModelCallback(BaseCallback[StepCallbackParams]):
    def __init__(self,
                 directory: Path,
                 save_frequency: int,
                 only_best: bool = True
                 ):
        self._directory = directory
        self._save_checkpoint = SaveCheckpointCallback(
            directory=self._directory
        )
        self._save_frequency = save_frequency
        self.only_best = only_best
        self._call_count_since_save = 0
        self._save_count = 0
        self._checkpoint_to_save: Optional[CheckpointCallbackParams] = None

    def __call__(self, params: CheckpointCallbackParams) -> None:
        if self.only_best:
            val_losses = params.ctx.losses[Train_Categories.Validation.value][Train_Features.Best.value]
            test_losses = params.ctx.losses[Train_Categories.Test.value][Train_Features.Best.value]
            losses = val_losses if len(val_losses) > len(test_losses) else test_losses
            should_save = self._save_count == 0 or len(losses) > 1 and losses[-1] < losses[-2]
        else:
            should_save = True

        if should_save:
            self._checkpoint_to_save = params
            self._save_count += 1

        if self._checkpoint_to_save is not None and self._call_count_since_save > self._save_frequency:
            self._save_checkpoint(self._checkpoint_to_save)
            self._call_count_since_save = 0
            self._checkpoint_to_save = None
        else:
            self._call_count_since_save += 1
