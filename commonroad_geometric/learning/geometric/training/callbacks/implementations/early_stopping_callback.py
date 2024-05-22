from typing import Dict, Optional

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, EarlyStoppingCallbacksParams
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features


class EarlyStoppingCallback(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(self, after_epochs: Optional[int] = None):
        self.after_epochs = after_epochs

    def __call__(self, params: EarlyStoppingCallbacksParams) -> Dict[str, bool]:
        if self.after_epochs is None or not params.ctx.losses[Train_Categories.Validation.value] or len(
                params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value]) == 0:
            return {type(EarlyStoppingCallback).__name__: False}
        best_epoch = min(range(len(params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value])),
                         key=params.ctx.losses[Train_Categories.Validation.value][Train_Features.Avg.value].__getitem__) + 1
        return {type(EarlyStoppingCallback).__name__: params.ctx.epoch - best_epoch >= self.after_epochs}
