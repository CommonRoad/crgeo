from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
import numpy as np
from tqdm import tqdm
import inspect
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
import torch.optim.lr_scheduler as lr_scheduler_module
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader  # DO not use from torch_geometric.loader import DataLoader, it will delete your collate_fn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Dataset
from torch_geometric.data.batch import Batch
from torch_geometric.loader.dataloader import Collater

from commonroad_geometric.common.progress_reporter import NoOpProgressReporter, ProgressReporter
from commonroad_geometric.common.utils.filesystem import save_dill
from commonroad_geometric.common.utils.system import get_gpu_count, get_gpu_usage
from commonroad_geometric.dataset.commonroad_dataset import T_Data
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import CheckpointCallbackParams, EarlyStoppingCallbacksParams, InitializeTrainingCallbacksParams, \
    InterruptCallbacksParams, LoggingCallbacksParams, StepCallbackParams
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputerContainerService, CallbackComputersContainer
from commonroad_geometric.learning.geometric.training.callbacks.implementations.early_stopping_callback import EarlyStoppingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.epoch_checkpoint_callback import EpochCheckpointCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.export_latest_model_callback import ExportLatestModelCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.gradient_clipping_callback import GradientClippingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.render_model import PYTHON_PATH_RENDER_MODEL, render_model
from commonroad_geometric.learning.geometric.training.training_utils import DebugCallback, custom_collate_fn
from commonroad_geometric.learning.geometric.training.types import GeometricTrainingContext, GeometricTrainingResults
from commonroad_geometric.learning.geometric.types import Train_Categories, Train_Features

if TYPE_CHECKING:
    from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
    from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
    from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment



def scalarize_and_convert_tensors(info_dict):
    def process_value(v):
        # Check if the item is a NumPy ndarray
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return v.item()  # Convert single-element arrays to scalars
            else:
                return v  # Return the array as is
        # Check if the item is a PyTorch tensor
        elif isinstance(v, torch.Tensor):
            if v.nelement() == 1 and v.ndim == 1:
                return v.item()  # Convert single-element tensors to scalars
            else:
                return v.detach().cpu().numpy()  # Detach, move to CPU, and convert to NumPy array
        else:
            return v  # Return the value unchanged if it's neither ndarray nor tensor

    # Apply the processing to each item in the dictionary
    return {k: process_value(v) for k, v in info_dict.items()}



ROLLING_MEAN_LOSS_COEFFICIENT = 0.99
ROLLING_MEAN_LOSS_COEFFICIENT_NEG = 1 - ROLLING_MEAN_LOSS_COEFFICIENT

@dataclass
class ComputedLosses:
    loss_dict: Tuple[List[float], List[Dict[str, Any]]]


@dataclass
class GeometricTrainerConfig:
    """Configuration options for trainer class.

        Args:
            output_dir (str):
                Directory where trained model is saved.
            multi_gpu (bool):
                Optional parameter signifying the use of multiple gpus if available (defaults to false).
            validation_freq (int):
                Optional paramer dictating the frequency with which validation should be performed (defaults to 1).
            enable_rendering (bool):
                Whether to render custom visualizations of model predictions during training
                from within a subprocess that continously loads the latest model checkpoint.
                Requires the 'configure_model' method to be implemented for the model.

            model_dir (str): Directory where trained model is saved.
            gpu_count (int, optional): Number of gpus. Defaults to 1.
            logger (Logger, optional): Logger class. Defaults to None.
            max_epochs (int, optional): Maximum epochs per training cycle. Defaults to 100.
            overfit (bool, optional): Overfit the model by training over 1 sample for max_epochs. Defaults to False.
            max_optimize_samples (int, optional): Maximum samples to be used for optimization if applicable. Defaults to 1000.
            validation_split (Union[float, int]): If float, the ratio between validation, train dataset samples. If integer, the number of validation samples. Defaults to 1.
    """

    backward_freq: int = 1
    batch_size: int = 16
    checkpoint_frequency: int = 100
    distributed_training: bool = False
    early_stopping: Optional[int] = None
    enable_multi_gpu: bool = False
    enable_rendering: bool = True
    gradient_clipping_threshold: Optional[float] = None
    log_freq: int = 100
    max_epochs: int = 100
    max_optimize_samples: int = 1
    overfit: bool = False
    render_subprocess: bool = True
    shuffle: bool = False
    swallow_errors: bool = False
    test_freq: Optional[int] = 1
    test_split: Union[float, int] = 0.1
    validate_inner: bool = True
    validation_freq: Optional[int] = 1
    validation_split: Union[float, int] = 1
    verbose: int = 1
    video_freq: int = 1000
    video_length: int = 400
    video_record_backoff: float = 1.5
    lr_scheduler_cls: Optional[str] = None
    lr_scheduler_kwargs: dict = field(default_factory=dict)


class GeometricTrainer:

    def __init__(
        self,
        *,
        cfg: GeometricTrainerConfig,
        device: str,
        logger: Optional[Logger] = None
    ) -> None:
        """Trainer for graph neural networks which extend the BaseGeometric class.

        Args:
            logger (Logger, optional): Logger class. Defaults to None.
        """

        self._cfg = cfg
        self._renderer_process_spawned: bool = False
        self._render_scenario_path: Optional[Path] = None
        self._model: Optional[BaseGeometric] = None
        self._gpu_count = get_gpu_count()
        self._device = device if device is not None else 'cuda' if self._gpu_count > 0 else 'cpu'
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized GeometricTrainer")

    @property
    def model(self) -> BaseGeometric:
        assert self._model is not None
        if self.multi_gpu():
            return self._model.module
        return self._model

    def multi_gpu(self) -> bool:
        device = self._device if isinstance(self._device, str) else self._device.type
        return self._gpu_count > 1 and device == 'cuda' and self._cfg.enable_multi_gpu

    def launch_trainer(self,
                       model_dir: Path,
                       experiment: GeometricExperiment,
                       dataset: CommonRoadDataset,
                       model: BaseGeometric,
                       wandb_service: Optional[WandbService] = None,
                       model_kwargs: Dict[str, Any] = {},
                       optimizer_service=None,
                       callbacks_computers: Optional[CallbackComputersContainer] = None,
                       wandb_experiment_args: Dict = None,
                       render_scenario_path: Optional[Path] = None,
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       collate_fn: Optional[Callable[[List[T_Data]], T_Data]] = None
                       ) -> None:
        self._render_scenario_path = render_scenario_path
        if collate_fn is None:
            parameters = inspect.signature(Collater.__init__).parameters
            requires_dataset = 'dataset' in parameters
            if requires_dataset:
                collate_fn = Collater(dataset=dataset, follow_batch=None, exclude_keys=None)
            else:
                collate_fn = Collater(follow_batch=None, exclude_keys=None)
        if self.multi_gpu():  # TODO create proper data handling instead of passing args everywhere
            args = (
                model_dir, experiment, dataset, model, wandb_service, model_kwargs,
                optimizer_service, optimizer_state, callbacks_computers, wandb_experiment_args,
                collate_fn
            )
            mp.spawn(self.load_dataset_and_setup_trainer, args=args, nprocs=self._gpu_count, join=True)
        else:
            self.load_dataset_and_setup_trainer(
                0, model_dir, experiment, dataset, model, wandb_service, model_kwargs,
                optimizer_service, optimizer_state, callbacks_computers, wandb_experiment_args, collate_fn
            )

    def load_dataset_and_setup_trainer(
        self,
        rank,
        model_dir: Path,
        experiment: GeometricExperiment,
        dataset: Dataset,
        model,
        wandb_service: WandbService,
        model_kwargs,
        optimizer_service,
        optimizer_state,
        callbacks_computers,
        wandb_experiment_args=None,
        collate_fn: Optional[Callable[[List[T_Data]], T_Data]] = None
    ):
        self._checkpoint_dir = model_dir.joinpath( "checkpoints")
        self._latest_dir = model_dir.joinpath("latest")

        if callbacks_computers is None:
            callbacks_computers = CallbackComputersContainer(
                training_step_callbacks=CallbackComputerContainerService([
                    ExportLatestModelCallback(
                        directory=self._latest_dir,
                        save_frequency=self._cfg.checkpoint_frequency if self._cfg.validate_inner else 1
                    ),
                    LogWandbCallback(wandb_service=wandb_service),
                    GradientClippingCallback(self._cfg.gradient_clipping_threshold)
                    # DebugTrainBackwardGradientsCallback(frequency=200)
                ]),
                validation_step_callbacks=CallbackComputerContainerService([
                    # LogInfoCallback()
                ]),
                logging_callbacks=CallbackComputerContainerService([LogWandbCallback(wandb_service=wandb_service)]),
                initialize_training_callbacks=CallbackComputerContainerService([WatchWandbCallback(
                    wandb_service=wandb_service,
                    log_freq=self._cfg.log_freq,
                    # log_gradients=not args.warmstart # TODO
                )]),
                checkpoint_callbacks=CallbackComputerContainerService([EpochCheckpointCallback(
                    directory=self._checkpoint_dir
                )]),
                early_stopping_callbacks=CallbackComputerContainerService([EarlyStoppingCallback(
                    after_epochs=self._cfg.early_stopping
                )]),
            )

        if self.multi_gpu():
            # Need to set master address and port for process spawning
            # see: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_batching.py
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=rank, world_size=self._gpu_count)
            if wandb_service is not None:
                wandb_service.start_experiment(**wandb_experiment_args)

        if optimizer_service is not None:
            subset = dataset.index_select(
                torch.arange(
                    len(dataset),
                    dtype=torch.long,
                    device="cpu")[
                    :self._cfg.max_optimize_samples])
            dataset_train, dataset_validation = subset[:int(
                self._cfg.max_optimize_samples / 2)], subset[int(self._cfg.max_optimize_samples / 2):]
        else:
            if self._cfg.overfit:
                dataset_train = dataset.index_select(slice(0, 1))
                dataset_test = dataset_train
                dataset_validation = dataset_train
            else:
                dataset_test, dataset_train = dataset.split(size=self._cfg.test_split)
                dataset_validation, dataset_train = dataset_train.split(size=self._cfg.validation_split)

        assert len(dataset_train) > 0

        if self._gpu_count > 0 and self._cfg.distributed_training:
            train_sampler = DistributedSampler(dataset_train, num_replicas=self._gpu_count, rank=rank)
            test_sampler = DistributedSampler(dataset_test, num_replicas=self._gpu_count, rank=rank)
            validation_sampler = DistributedSampler(dataset_validation, num_replicas=self._gpu_count, rank=rank)
            train_loader = DataLoader(
                dataset_train,
                batch_size=self._cfg.batch_size,
                sampler=train_sampler,
                #collate_fn=custom_collate_fn
            )
            test_loader = DataLoader(
                dataset_test,
                batch_size=self._cfg.batch_size,
                sampler=test_sampler,
                #collate_fn=custom_collate_fn
            )
            validation_loader = DataLoader(
                dataset_validation,
                batch_size=self._cfg.batch_size,
                sampler=validation_sampler,
                #collate_fn=custom_collate_fn
            )
        else:
            # loader = DataLoader(
            #     dataset,
            #     batch_size=self._cfg.batch_size,
            #     collate_fn=custom_collate_fn,
            #     shuffle=self._cfg.shuffle
            # )
            # TODO TODO TODO TODO
            train_loader = DataLoader(
                dataset_train,
                batch_size=self._cfg.batch_size,
                collate_fn=custom_collate_fn,
                shuffle=self._cfg.shuffle
            )
            test_loader = DataLoader(
                dataset_test,
                batch_size=self._cfg.batch_size,
                collate_fn=custom_collate_fn,
                shuffle=self._cfg.shuffle
            )
            validation_loader = DataLoader(
                dataset_validation,
                batch_size=self._cfg.batch_size,
                collate_fn=custom_collate_fn,
                shuffle=self._cfg.shuffle
            )

        self._train_loader = train_loader
        self._test_loader = test_loader
        self._val_loader = validation_loader

        if self._render_scenario_path is None:
            self._render_scenario_path = next(
                f for f in test_loader.dataset.raw_paths if test_loader.dataset[0].scenario_id[0] in f)
        self._callbacks_computers = callbacks_computers
        self.logger.info(f"Model: {model}")

        results = self.train_orchestrator(
            experiment=experiment,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            validation_loader=validation_loader,
            device=rank if self.multi_gpu() else self._device,
            optimizer_service=optimizer_service,
            optimizer_state=optimizer_state,
            **model_kwargs
        )
        if wandb_service is not None:
            wandb_service.log(vars(results))

        if optimizer_service is not None:
            optimizer_service.conclude_trial()

        if self._gpu_count > 1:
            dist.destroy_process_group()

    def train_orchestrator(
        self,
        experiment: GeometricExperiment,
        model: BaseGeometric,
        train_loader: DataLoader,
        test_loader: DataLoader,
        validation_loader: DataLoader,
        device: Union[str, torch.device] = 'cpu',
        optimizer = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> GeometricTrainingResults:
        """Trains the graph neural network.

        Args:
            train_loader (DataLoader):
                DataLoader used for training.
            validation_loader (DataLoader):
                DataLoader used for computing the validation loss.
            max_epochs (int, optional):
                Maximum epochs to train for or 0 to train until the training is stopped. Defaults to 0.
            device (str | torch.device, optional):
                PyTorch device. Defaults to 'cpu'.

        Returns:
            TrainingResults:
                Result object including key metrics for the training run.
        """

        device = torch.device(device)

        if len(self._train_loader.dataset.raw_paths) == 0:
            raise ValueError(f"Train dataset '{self._train_loader.dataset.raw_dir} has no data'")

        max_epochs = int(self._cfg.max_epochs)
        epochs = range(1, max_epochs + 1) if max_epochs > 0 else itertools.count(start=1)
        progress_cls = ProgressReporter if self._cfg.verbose > 0 else NoOpProgressReporter
        progress_kwargs = dict(report_memory=True) if self._cfg.verbose > 0 else {}
        progress = progress_cls(total=max_epochs, unit="epoch", **progress_kwargs)
        self._optimizer_service = optimizer

        if self._optimizer_service is not None:
            self._optimizer_service.optimize(
                self.train, model, experiment, train_loader, test_loader,
                validation_loader, device, kwargs, epochs, progress
            )
        else:
            self.train(None, model, experiment, train_loader, test_loader,
                       validation_loader, device, kwargs, epochs, progress, optimizer_state=optimizer_state
                       )

        debug_callback = DebugCallback()
        results = GeometricTrainingResults(
            epochs=self._ctx.epoch,
            losses=debug_callback(self._ctx),
            duration=time.time() - self._ctx.start_time,
        )

        self.logger.info("Training completed")

        return results

    def validate(
        self,
        model: BaseGeometric,
        data: Batch,
        device: torch.device,
        raise_errors: bool = True,
        is_test: bool = False,
        ctx: Optional[GeometricTrainingContext] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        return_outputs: bool = False
    ) -> Tuple[List[Tensor], Optional[float], Dict[str, float]]:
        model.train(False)
        with torch.no_grad():
            try:
                val_outputs, validation_losses, val_info_dicts = self.compute_loss(
                    model=model,
                    device=device,
                    ctx=ctx,
                    loader=data,
                    val=True,
                    raise_errors=raise_errors,
                    progress_reporter=progress_reporter,
                    return_outputs=return_outputs
                )
            except Exception as e:
                if raise_errors:
                    raise
                self.logger.error(e, exc_info=True)
                return None, {}
        avg_validation_loss = sum(validation_losses) / len(validation_losses) if len(validation_losses) else None
        info_dict: Dict[str, float] = {}
        if len(validation_losses) > 0:
            # computing averages
            for key in val_info_dicts[0]:
                info_dict[key] = 0.0
            for sub_info_dict in val_info_dicts:
                for key, value in sub_info_dict.items():
                    try:
                        info_dict[key] += value
                    except RuntimeError:
                        info_dict[key] = value  # TODO
            for key in info_dict:
                info_dict[key] /= len(validation_losses)

        if ctx is not None:
            category_key = Train_Categories.Test.value if is_test else Train_Categories.Validation.value
            ctx.losses[category_key][Train_Features.Avg.value].append(avg_validation_loss)
            ctx.info_dict[category_key].append(deepcopy(info_dict))
            if not ctx.losses[category_key][Train_Features.Best.value]:
                ctx.losses[category_key][Train_Features.Best.value].append(avg_validation_loss)
            elif avg_validation_loss < ctx.losses[category_key][Train_Features.Best.value][-1]:
                ctx.losses[category_key][Train_Features.Best.value].append(avg_validation_loss)
            else:
                ctx.losses[category_key][Train_Features.Best.value].append(
                    ctx.losses[category_key][Train_Features.Best.value][-1])

        return val_outputs, avg_validation_loss, info_dict

    def train(
        # TODO: Type hints
        self,
        trial,
        model: BaseGeometric,
        experiment: GeometricExperiment,
        train_loader: DataLoader,
        test_loader: DataLoader,
        validation_loader: DataLoader,
        device,
        kwargs,
        epochs,
        progress,
        optimizer_state: Optional[Dict[str, Any]]
    ):
        if len(train_loader) == 0:
            raise ValueError("Cannot train model with empty train loader")

        self._model = model
        self._experiment = experiment
        if experiment._config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

        build_batch = next(iter(train_loader)).to(device)
        validation_batch = next(iter(validation_loader)).to(device)
        self._model.build(
            build_batch,
            trial,
            optimizer_state=optimizer_state,
            **kwargs)
        self._model.to(device)
        if self.multi_gpu():
            from torch.nn.parallel import DistributedDataParallel
            self._model.to(device)
            self._model = DistributedDataParallel(self._model, device_ids=[device])

        Train_Features.__getitem__('Best')
        self._ctx = GeometricTrainingContext(
            device=device,
            model=self.model,
            optimizer=self.model.optimizer,
            start_time=time.time(),
            epoch=-1,
            step=-1,
            losses=dict(
                train={Train_Features.__getitem__(i).value: [] for i in Train_Features._member_names_},
                test={Train_Features.__getitem__(i).value: [] for i in Train_Features._member_names_},
                validation={Train_Features.__getitem__(i).value: [] for i in Train_Features._member_names_}
            ),
            info_dict=dict(
                train=[],
                test=[],
                validation=[]
            )
        )

        self.logger.info(f"Starting training on {device}. Train/test set size: {len(train_loader)}/{len(test_loader)}. Batch size: {train_loader.batch_size}")
        scheduler: Optional[_LRScheduler] = None
        if self._cfg.lr_scheduler_cls is not None:
            scheduler_cls = getattr(lr_scheduler_module, self._cfg.lr_scheduler_cls)
            scheduler = scheduler_cls(self.model.optimizer, **self._cfg.lr_scheduler_kwargs)

        self._callbacks_computers.initialize_training_callbacks(InitializeTrainingCallbacksParams(ctx=self._ctx))
        try:
            for epoch in epochs:
                self._ctx.epoch = epoch
                progress.update(epoch)

                self.model.train(True)
                train_losses, train_info_dicts = self.train_loop(
                    ctx=self._ctx,
                    loader=train_loader,
                    validation_batch=validation_batch,
                    parent_progress_reporter=progress,
                    scheduler=scheduler
                )
                avg_train_loss = sum(train_losses) / len(train_losses)
                self._ctx.losses[Train_Categories.Train.value][Train_Features.Avg.value].append(avg_train_loss)
                self._ctx.info_dict[Train_Categories.Train.value].append(train_info_dicts)

                logging_callback_dict = {"epoch": epoch, "train_losses": train_losses}

                if not self._cfg.validate_inner and self._cfg.validation_freq is not None and (
                        epoch + 1) % self._cfg.validation_freq == 0:
                    val_outputs, val_loss, val_info_dicts = self.validate(
                        model=self.model,
                        data=validation_loader,
                        device=self._ctx.device,
                        ctx=self._ctx,
                        raise_errors=not self._cfg.swallow_errors,
                        return_outputs=True
                    )
                    if self._callbacks_computers is not None and self._callbacks_computers.validation_step_callbacks is not None:
                        self._callbacks_computers.validation_step_callbacks(StepCallbackParams(
                            ctx=self._ctx,
                            output=val_outputs[0],
                            train_loss=val_loss,
                            info_dict=val_info_dicts[0] if 0 in val_info_dicts else None,
                            batch=validation_batch,
                        ))

                    logging_callback_dict["val_loss"] = f"{val_loss:.3f} ({self._ctx.losses[Train_Categories.Validation.value][Train_Features.Best.value][-1]:.3f})"

                if self._cfg.test_freq is not None and (epoch + 1) % self._cfg.test_freq == 0:
                    _, test_loss, test_info_dicts = self.validate(
                        model=self.model,
                        data=test_loader,
                        device=self._ctx.device,
                        ctx=self._ctx,
                        is_test=True,
                        raise_errors=not self._cfg.swallow_errors,
                        return_outputs=False
                    )
                    if self._callbacks_computers is not None and self._callbacks_computers.test_step_callbacks is not None:
                        self._callbacks_computers.test_step_callbacks(StepCallbackParams(
                            ctx=self._ctx,
                            output=None,
                            train_loss=test_loss,
                            info_dict=test_info_dicts,
                            batch=test_loader.dataset[0],
                        ))
                    logging_callback_dict["test_loss"] = f"{test_loss:.3f} ({self._ctx.losses[Train_Categories.Test.value][Train_Features.Best.value][-1]:.3f})"
                logging_callback_dict['lr'] = self._ctx.optimizer.param_groups[0]

                self.logger.debug(f"Completed epoch {epoch}/{epochs}. GPU usage is {get_gpu_usage():.2%}")

                if self._callbacks_computers.logging_callbacks is not None:
                    self._callbacks_computers.logging_callbacks(
                        LoggingCallbacksParams(ctx=self._ctx, kwargs=logging_callback_dict))

                if self._callbacks_computers.early_stopping_callbacks is not None:
                    early_stop = self._callbacks_computers.early_stopping_callbacks(
                        EarlyStoppingCallbacksParams(ctx=self._ctx))
                    if type(EarlyStoppingCallback).__name__ in early_stop and early_stop[type(
                            EarlyStoppingCallback).__name__]:
                        break
                # Add callbacks for checkpointing
                if self._callbacks_computers.checkpoint_callbacks is not None:
                    self._callbacks_computers.checkpoint_callbacks(CheckpointCallbackParams(ctx=self._ctx))

                if self._optimizer_service is not None:
                    self._optimizer_service.prune_trial(trial, epoch, self._ctx)

                if self._cfg.enable_rendering and not self._renderer_process_spawned and not self._cfg.render_subprocess:
                    self._spawn_render_process()

        except KeyboardInterrupt:
            if self._callbacks_computers.interrupt_callbacks is not None:
                self._callbacks_computers.interrupt_callbacks(InterruptCallbacksParams(ctx=self._ctx))
        finally:
            progress.close()

        if self._optimizer_service is not None:
            return self._optimizer_service.get_metrics(self._ctx)

    def base_loop(func):
        def wrapper(*args, **kwargs):
            if 'parent_progress_reporter' in kwargs:
                try:
                    total = int(kwargs['loader'].sampler.num_samples / kwargs['loader'].batch_size)
                except AttributeError:
                    total = len(kwargs['loader'])
                kwargs['progress_reporter'] = type(kwargs['parent_progress_reporter'])(
                    total=total,
                    unit='b',
                    parent_reporter=kwargs['parent_progress_reporter'] if 'parent_progress_reporter' in kwargs else None
                )
            return func(*args, **kwargs)
        return wrapper

    @base_loop
    def train_loop(
        self,
        *,
        ctx: GeometricTrainingContext,
        loader: DataLoader,
        validation_batch: Batch,
        parent_progress_reporter: ProgressReporter = None,
        progress_reporter: ProgressReporter = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> Tuple[List[float], List[Dict[str, Tensor]]]:
        losses: List[float] = []
        info_dicts: List[Dict[str, Tensor]] = []
        backward_count: int = 0
        train_loss_sum: float = 0.0
        train_loss_mean: float

        ctx.optimizer.zero_grad()
        ctx.model.train(True)

        for index, batch in enumerate(loader):
            # if loader.batch_size != batch.batch_size:
            #     continue

            self._ctx.step += 1

            batch = batch.to(self._device)

            def execute_step() -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
                nonlocal ctx
                nonlocal batch
                batch = ctx.model.train_preprocess(batch)
                output = ctx.model(batch)
                loss, info_dict = ctx.model.compute_loss(output, batch)
                info_dict = scalarize_and_convert_tensors(info_dict)
                return output, loss, info_dict

            if self._cfg.swallow_errors:
                try:
                    train_output, train_loss_step, info_dict = execute_step()
                except Exception as e:
                    if not self._cfg.swallow_errors:
                        raise e
                    self.logger.error(e, exc_info=True)
                    continue
            else:
                train_output, train_loss_step, info_dict = execute_step()

            info_dict = scalarize_and_convert_tensors(info_dict)
            train_loss_step.backward()

            train_loss_sum += train_loss_step.item()

            if index % self._cfg.backward_freq == 0:
                train_loss = train_loss_sum / self._cfg.backward_freq
                backward_count += 1

                if self._cfg.validate_inner and self._cfg.validation_freq is not None and backward_count % self._cfg.validation_freq == 0:
                    val_outputs, val_loss, val_info_dicts = self.validate(
                        model=self.model,
                        data=validation_batch,
                        device=self._ctx.device,
                        ctx=self._ctx,
                        raise_errors=not self._cfg.swallow_errors,
                        return_outputs=True
                    )
                    if self._callbacks_computers is not None and self._callbacks_computers.validation_step_callbacks is not None:
                        self._callbacks_computers.validation_step_callbacks(StepCallbackParams(  # TODO
                            ctx=ctx,
                            output=val_outputs[0],
                            train_loss=val_loss,
                            info_dict=val_info_dicts[0] if 0 in val_info_dicts else None,
                            batch=validation_batch,
                        ))
                    if len(self._ctx.losses[Train_Categories.Validation.value][Train_Features.Best.value]) > 0:
                        info_dict["vl"] = f"{val_loss:.3f} ({self._ctx.losses[Train_Categories.Validation.value][Train_Features.Best.value][-1]:.3f})"
                    self.model.train(True)

                ctx.losses[Train_Categories.Train.value][Train_Features.Current.value].append(train_loss)
                losses.append(train_loss)
                info_dicts.append(info_dict)

                if self._callbacks_computers is not None and self._callbacks_computers.training_step_callbacks is not None:
                    self._callbacks_computers.training_step_callbacks(StepCallbackParams(
                        ctx=ctx,
                        train_loss=train_loss,
                        info_dict=info_dict,
                        batch=batch,
                        output=train_output
                    )
                    )

                ctx.optimizer.step()
                ctx.optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                if index == 0:
                    train_loss_mean = train_loss
                else:
                    train_loss_mean = ROLLING_MEAN_LOSS_COEFFICIENT * train_loss_mean + ROLLING_MEAN_LOSS_COEFFICIENT_NEG * train_loss

                progress_metrics = {}
                progress_metrics.update({'tl': f"{train_loss:.3f} ({train_loss_mean:.3f})",
                                        'lr': self._ctx.optimizer.param_groups[0]['lr']})
                progress_metrics.update(info_dict)
                if progress_reporter is not None:
                    progress_reporter.set_postfix_str(', '.join([f'{k}: ' + ((f'{v:.3f}' if v > 1e-3 else f'{v:.2e}') if isinstance(
                        v, float) else v) for k, v in progress_metrics.items() if isinstance(v, (float, str))]))
                    progress_reporter.update(index)
                train_loss_sum = 0.0

                if self._cfg.verbose:
                    tqdm.write(", ".join(f"{k + ':'} {v:.4f}" for k, v in info_dict.items() if isinstance(v, float)))

                if self._cfg.enable_rendering and not self._renderer_process_spawned and self._cfg.render_subprocess and self._ctx.model.latest_model_path is not None:
                    self._spawn_render_process()
                    self._renderer_process_spawned = True

        if progress_reporter is not None:
            progress_reporter.close()
        return losses, info_dicts

    @base_loop
    def compute_loss(
        self,
        model: BaseGeometric,
        loader: Union[DataLoader, Batch],
        device: torch.device,
        val: bool = False,
        parent_progress_reporter: Optional[ProgressReporter] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        ctx: Optional[GeometricTrainingContext] = None,
        raise_errors: bool = True,
        return_outputs: bool = False
    ) -> Tuple[List[Tensor], List[float], List[Dict[str, Tensor]]]:

        outputs: List[Tensor] = []
        losses: List[float] = []
        info_dicts: List[Dict[str, Tensor]] = []

        if not isinstance(loader, DataLoader):
            loader = [loader]

        for index, batch in enumerate(loader):
            batch = batch.to(device)

            try:
                batch = model.train_preprocess(batch)
                output = model.forward(batch)
                loss_th, info_dict = model.compute_loss(output, batch)
            except Exception as e:
                if raise_errors:
                    raise
                self.logger.error(e, exc_info=True)
                continue

            loss = loss_th.item()
            if ctx is not None:
                if val:
                    ctx.losses[Train_Categories.Validation.value][Train_Features.Current.value].append(loss)
                else:
                    ctx.losses[Train_Categories.Test.value][Train_Features.Current.value].append(loss)
            if return_outputs:
                outputs.append(output)
            losses.append(loss)
            info_dict = scalarize_and_convert_tensors(info_dict)
            info_dicts.append(info_dict)
            if progress_reporter is not None:
                progress_reporter.update(index)
        if progress_reporter is not None:
            progress_reporter.close()

        return outputs, losses, info_dicts

    def _spawn_render_process(
        self
    ) -> None:
        renderer_plugins = self.model.configure_renderer_plugins()
        if renderer_plugins is None:
            return

        video_dir = os.path.abspath(os.path.join(self._latest_dir, 'videos'))

        if self._cfg.render_subprocess:
            model_path = self._ctx.model.latest_model_path
            assert model_path is not None
            render_plugins_path = os.path.abspath(os.path.join(self._latest_dir, 'render_plugins'))
            save_dill(renderer_plugins, render_plugins_path)
            experiment_path = os.path.abspath(self._experiment.save(str(self._latest_dir)))
            cmd = f'{sys.executable} "{PYTHON_PATH_RENDER_MODEL}" ' + \
                f'--cwd "{os.getcwd()}" ' + \
                f'--scenario "{self._render_scenario_path}" ' + \
                f'--experiment "{experiment_path}" ' + \
                f'--model-dir "{model_path}" ' + \
                f'--plugins "{render_plugins_path}" ' + \
                f'--video-dir "{video_dir}" ' + \
                f'--video-length {self._cfg.video_length} ' + \
                f'--video-freq {self._cfg.video_freq} ' + \
                f'--record-backoff {self._cfg.video_record_backoff}'

            subprocess.Popen(cmd, shell=True)
            self.logger.info(f"Spawned rendering subprocess {PYTHON_PATH_RENDER_MODEL}")
            self.logger.debug(f"Executed command was: {cmd}")

        else:
            render_model(
                model=self._ctx.model,
                experiment=self._experiment,
                scenario_path=self._render_scenario_path,
                renderer_plugins=renderer_plugins,
                video_dir=video_dir,
                video_freq=self._cfg.video_freq,
                video_length=self._cfg.video_length,
                video_record_backoff=self._cfg.video_record_backoff
            )
