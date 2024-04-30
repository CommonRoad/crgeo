from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Generic, List

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import (BaseCallback, CheckpointCallbackParams, EarlyStoppingCallbacksParams, InitializeTrainingCallbacksParams,
                                                                                      InterruptCallbacksParams, LoggingCallbacksParams, StepCallbackParams, TypeVar_CallbackParams)


class CallbackComputerContainerService(Generic[TypeVar_CallbackParams]):
    """ Gets data from git and sends it over to git feature computers to get a dictionary of relevant information
    """

    def __init__(
        self,
        callbacks: List[BaseCallback[TypeVar_CallbackParams]]
    ) -> Dict:
        self._callbacks = callbacks

    def __call__(self, params: TypeVar_CallbackParams) -> Dict:
        computations: Dict = {}
        for callback in self._callbacks:
            callback_return = callback(params=params)
            if callback_return is not None:
                computations.update(callback_return)
        return computations


@dataclass
class CallbackComputersContainer(ABC):
    logging_callbacks: CallbackComputerContainerService[LoggingCallbacksParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    training_step_callbacks: CallbackComputerContainerService[StepCallbackParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    validation_step_callbacks: CallbackComputerContainerService[StepCallbackParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    test_step_callbacks: CallbackComputerContainerService[StepCallbackParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    initialize_training_callbacks: CallbackComputerContainerService[InitializeTrainingCallbacksParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    interrupt_callbacks: CallbackComputerContainerService[InterruptCallbacksParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    early_stopping_callbacks: CallbackComputerContainerService[EarlyStoppingCallbacksParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
    checkpoint_callbacks: CallbackComputerContainerService[CheckpointCallbackParams] = field(
        default_factory=lambda: CallbackComputerContainerService([]))
