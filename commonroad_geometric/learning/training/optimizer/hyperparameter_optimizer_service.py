from __future__ import annotations

import logging
from typing import Any, Callable, List, TYPE_CHECKING, Union

import optuna
import torch.optim as optim
from optuna import Study, Trial, pruners
from optuna.pruners._hyperband import HyperbandPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState

from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.learning.training.optimizer.weights_and_biases_callback import WeightsAndBiasesCallback

if TYPE_CHECKING:
    from commonroad_geometric.learning.training.wandb_service import WandbService

logger = logging.getLogger(__name__)


class BaseOptimizerService:
    def __init__(
        self,
        metrics: List[str] = None,
        **kwargs
    ) -> None:
        ...

    def get_metrics(self, *args, **kwargs) -> Any:
        ...

    def optimize(self, objective: Callable, *args, **kwargs):
        ...

    def prune_trial(self, trial, epoch: int, *args, **kwargs):
        ...

    def suggest_optimizer(parameters, *args, **kwargs):
        ...
    
    def suggest_param(func_name: str, attr: str, *args, **kwargs):
        ...

    def conclude_trial(self) -> None:
        ...


# Based on wandb sweeps
class SweepsOptimizerService(BaseOptimizerService):
    def __init__(
        self,
        wandb_service: WandbService = None,
        metrics: Union[str, List[str]] = None,
        metric_callback: Callable[..., Any] = None,
        log_file: str = 'optimizer.log',
    ) -> None:
        # Use directions for mult-parameter learning
        self._metric_callback = metric_callback
        self._metrics = metrics
        self._wandb_service = wandb_service
        setup_logging(filename=log_file)

    @property
    def wandb_service(self) -> WandbService:
        return self._wandb_service

    def get_metrics(self, *args, **kwargs) -> Any:
        return self._metric_callback(*args, **kwargs)

    def optimize(self, objective: Callable, *args, **kwargs):
        func = lambda trial: objective(trial, *args, **kwargs)
        return func(self._wandb_service.config.as_dict())

    def prune_trial(self, *args, **kwargs):
        ...

    def suggest_optimizer(self,
        parameters,
        *args,
        **kwargs,
    ):
        print(parameters)
        lr = self.wandb_service.config.learning_rate if 'learning_rate' in self.wandb_service.config else 1e-3
        optimizer = self.wandb_service.config.optimizer if 'optimizer' in self.wandb_service.config else 'Adam'
        return getattr(optim, optimizer)(parameters, lr=lr)

    def suggest_param(self, func_name: str, attr: str, *args, **kwargs):
        return self.wandb_service.config[attr]

    def conclude_trial(self) -> None:
        ...


class OptunaOptimizerService(BaseOptimizerService):
    def __init__(
        self,
        directions: Union[str, List[str]] = 'maximize',
        sampler: BaseSampler = TPESampler(),
        wandb_service: WandbService = None,
        n_trials=100,
        metrics: Union[str, List[str]] = None,
        metric_callback: Callable[..., Any] = None,
        log_file: str = 'optimizer.log',
        optimizer_suggestions: List[str] = ["Adam", "RMSprop", "SGD"],
        pruner: pruners.BasePruner = None,
        early_stopping: bool = True
    ) -> None:
        self._multi_parameter = isinstance(directions, list)
        # Use directions for mult-parameter learning
        if pruner is None:
            pruner = HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3)
        self._study = optuna.create_study(sampler=sampler, pruner=pruner, direction=None if self._multi_parameter else directions, directions=directions if self._multi_parameter else None)
        self._metric_callback = metric_callback
        self._metrics = metrics
        self._callbacks = []
        self._n_trials = n_trials
        self._optimizer_suggestions = optimizer_suggestions
        self._early_stopping = early_stopping
        setup_logging(filename=log_file)
        if wandb_service is not None:
            self._wandb_service = wandb_service
            self._wandbc = WeightsAndBiasesCallback(metric_name=metrics, wandb_service=self._wandb_service)
            self._callbacks.append(self._wandbc)

    @property
    def metrics(self) -> Union[str, List[str], None]:
        return self._metrics

    @property
    def study(self) -> Study:
        return self._study

    @property
    def multi_parameter(self) -> bool:
        return self._multi_parameter

    @property
    def wandb_service(self) -> WandbService:
        return self._wandb_service

    def get_metrics(self, *args, **kwargs) -> Any:
        return self._metric_callback(*args, **kwargs)

    def optimize(self, objective: Callable, *args, **kwargs):
        func = lambda trial: objective(trial, *args, **kwargs)
        self._study.optimize(func, n_trials=self._n_trials, callbacks=self._callbacks)

    def prune_trial(self, trial, epoch, *args, **kwargs):
        if not self._early_stopping:
            return
        if not self._multi_parameter:
            trial.report(self.get_metrics(*args, **kwargs), epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    def suggest_optimizer(self,
        parameters,
        trial: Trial,
        lr_high:float = 1e-1, 
        lr_low:float = 1e-5
    ):
        print(parameters)
        optimizer_name = trial.suggest_categorical("optimizer", self._optimizer_suggestions)
        lr = trial.suggest_float("lr", lr_low, lr_high)
        return getattr(optim, optimizer_name)(parameters, lr=lr)

    def suggest_param(self, func_name: str, attr:str, trial: optuna.Trial, *args, **kwargs):
        return getattr(trial, func_name)(attr, *args, **kwargs)

    def conclude_trial(self) -> None:

        pruned_trials = self._study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self._study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info(f'  Number of finished trials: {len(self._study.trials)}')
        logger.info(f' Number of pruned trials: {len(pruned_trials)}')
        logger.info(f' Number of complete trials: {len(complete_trials)}')

        logger.info("Best trial:")
        trial = self._study.best_trial
        logger.info(f'  Value: {trial.value}')
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f'    {key}: {value}')
