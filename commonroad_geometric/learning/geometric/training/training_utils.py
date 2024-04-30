from math import isfinite
from typing import Any, Callable, Dict, List, Optional, TypeVar

from torch_geometric.data import Batch, Data, HeteroData

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal, CommonRoadDataTemporalBatch
from commonroad_geometric.learning.geometric.training.types import GeometricTrainingCallback, GeometricTrainingContext
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features


def collect_return_values(callbacks: Dict[str, Callable]) -> Callable:
    """Run the callbacks sequentially (in insertion order) and
    provide each callback an additional keyword argument `return_values`
    which is a dictionary containing the return values of previous
    callbacks.

    The return value of the last callback is returned to the caller.
    """

    if not callbacks:
        raise ValueError("The callbacks dictionary must not be empty")

    def wrapper_fn(*args, **kwargs):
        return_values = {}
        for name, callback in callbacks.items():
            val = callback(*args, **kwargs, return_values=return_values)
            return_values[name] = val
        return val

    return wrapper_fn


class DebugCallback(GeometricTrainingCallback):
    def __init__(self):
        self.running_avg_train_loss: Optional[float] = None

    def __call__(self, ctx: GeometricTrainingContext, *args) -> Dict[str, Any]:
        # Updating running average for training loss
        train_loss = ctx.losses[Train_Categories.Train.value][Train_Features.Current.value][-1]
        if self.running_avg_train_loss is None:
            self.running_avg_train_loss = train_loss
        else:
            self.running_avg_train_loss = 0.99 * self.running_avg_train_loss + 0.01 * train_loss

        callback_dict = {Train_Categories.__getitem__(i).value + '_' + Train_Features.__getitem__(j).value: self.get_loss_value(ctx, i, j)[-1] if len(self.get_loss_value(ctx, i, j)) > 0 and isfinite(self.get_loss_value(ctx, i, j)[-1]) else None
                         for i in Train_Categories._member_names_ if Train_Categories.__getitem__(i).value in ctx.losses for j in Train_Features._member_names_}

        callback_dict['step'] = ctx.step
        callback_dict['epoch'] = ctx.epoch
        return callback_dict

    def get_loss_value(self, ctx, i, j):
        return ctx.losses[Train_Categories.__getitem__(i).value][Train_Features.__getitem__(j).value]


class OptimizationMetricsCallback(GeometricTrainingCallback):
    def __init__(self):
        ...

    def __call__(self, ctx: GeometricTrainingContext, *args) -> None:
        if not ctx.losses[Train_Categories.Validation.value] or len(
                ctx.losses[Train_Categories.Validation.value][Train_Features.Current.value]) == 0:
            return False
        return ctx.losses[Train_Categories.Validation.value][Train_Features.Current.value][-1]


T_Data = TypeVar("T_Data", Data, HeteroData, CommonRoadData, CommonRoadDataTemporal)
T_Batch = TypeVar("T_Batch", Batch, CommonRoadDataTemporalBatch)


def custom_collate_fn(data_list: List[T_Data], follow_batch=None, exclude_keys=None) -> T_Batch:
    """
    creates assignment vectors for each key in :obj:`follow_batch`.
    Will exclude any keys given in :obj:`exclude_keys`.
    """
    is_cr_data_temporal = isinstance(data_list[0], CommonRoadDataTemporal)
    if is_cr_data_temporal:
        batch = CommonRoadDataTemporalBatch.from_data_list(data_list, follow_batch, exclude_keys)
    else:
        batch = Batch.from_data_list(data_list, follow_batch, exclude_keys)

    return batch
