from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, BaseCallbackParams, InitializeTrainingCallbacksParams
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService


class WatchWandbCallback(BaseCallback[InitializeTrainingCallbacksParams]):
    def __init__(
        self,
        wandb_service: WandbService,
        log_freq: int = 100,
        log_gradients: bool = True
    ):
        self._wandb_service = wandb_service
        self._log_freq = log_freq
        self._log_gradients = log_gradients

    def __call__(
        self,
        params: BaseCallbackParams,
    ):
        if self._wandb_service.success:
            # Logging gradients can randomly fail with warmstart, not sure why
            # https://github.com/wandb/client/issues/800
            self._wandb_service.watch_model(
                params.ctx,
                log="all" if self._log_gradients else "parameters",
                log_freq=self._log_freq)
