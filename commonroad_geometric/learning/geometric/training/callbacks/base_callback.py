from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, TypeVar, Union

from torch import Tensor

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.training.training_utils import GeometricTrainingContext


@dataclass
class BaseCallbackParams:
    ctx: GeometricTrainingContext


@dataclass
class StepCallbackParams(BaseCallbackParams):
    ctx: GeometricTrainingContext
    train_loss: Union[Tensor, float]
    batch: CommonRoadData
    info_dict: Dict[str, Any]
    output: Tensor


@dataclass
class InitializeTrainingCallbacksParams(BaseCallbackParams):
    ...


@dataclass
class LoggingCallbacksParams(BaseCallbackParams):
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]
    ...


@dataclass
class EarlyStoppingCallbacksParams(BaseCallbackParams):
    ...


@dataclass
class InterruptCallbacksParams(BaseCallbackParams):
    ...


@dataclass
class CheckpointCallbackParams(BaseCallbackParams):
    ...    


TypeVar_CallbackParams = TypeVar(
    "TypeVar_CallbackParams",
    bound=Union['BaseCallbackParams', Callable],
)


class BaseCallback(Generic[TypeVar_CallbackParams], ABC, AutoReprMixin):
    """
    Base class for custom callback implementations
    dictionary
    """

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(
        self,
        params: TypeVar_CallbackParams,
    ) -> Union[None, Dict]:
        ...
