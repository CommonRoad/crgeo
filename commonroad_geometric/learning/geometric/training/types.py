from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar, Union

import torch
from torch.optim import Optimizer

from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric


class GeometricTrainingCallback(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


T_GeometricTrainingCallback = TypeVar("T_GeometricTrainingCallback", Callable, GeometricTrainingCallback)
TrainingCallbacks = Dict[str, Union[T_GeometricTrainingCallback, List[T_GeometricTrainingCallback]]]


@dataclass
class GeometricTrainingContext:
    device: torch.device
    model: BaseGeometric
    optimizer: Optimizer
    start_time: float
    step: int
    epoch: int
    losses: Dict[str, List[float]]
    info_dict: Dict[str, Any]


@dataclass(frozen=True)
class GeometricTrainingResults:
    epochs: int
    duration: float
    losses: Dict[str, List[float]]
