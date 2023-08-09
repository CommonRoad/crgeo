from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, BaseCallbackParams
from commonroad_geometric.learning.geometric.training.training_utils import DebugCallback
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService


class LogWandbCallback(BaseCallback[BaseCallbackParams]):
    def __init__(self, wandb_service: WandbService):
        self._wandb_service = wandb_service
        self._metrics_initialized = False

    def __call__(
        self,
        params: BaseCallbackParams,
    ):
        self.initialize_metrics()

        if self._wandb_service.success:
            debug_callback = DebugCallback()
            metrics = debug_callback(params.ctx)
            self._wandb_service.log(metrics)
            if hasattr(params, 'info_dict') and params.info_dict is not None:
                self._wandb_service.log(params.info_dict)

    def initialize_metrics(self):
        if not self._metrics_initialized:
            self._wandb_service.define_metric('step')
            for i in Train_Categories._member_names_:
                for j in Train_Features._member_names_:
                    self._wandb_service.define_metric(
                        Train_Categories.__getitem__(i).value + '_' + Train_Features.__getitem__(j).value, step_metric='step'
                    )
            self._metrics_initialized = True
