from typing import Optional
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, CheckpointCallbackParams, EarlyStoppingCallbacksParams
import torch

class GradientClippingCallback(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(self, gradient_threshold: Optional[float] = None):
        self.gradient_threshold = gradient_threshold

    def __call__(self, params: CheckpointCallbackParams) -> None:
        if self.gradient_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                params.ctx.model.parameters(),
                self.gradient_threshold
            )
