from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import os
import shutil
import torch
import logging
from commonroad_geometric.common.config import Config
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.common.utils.string import filter_alpha
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory
from commonroad_geometric.debugging.profiling import profile
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import get_most_recent_file, list_files, load_dill, search_file
from commonroad_geometric.common.utils.seeding import set_global_seed
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset, CommonRoadDatasetMissingException, T_Data
from commonroad_geometric.learning.base_project import BaseProject, register_run_command
from commonroad_geometric.learning.geometric.training.render_model import render_model
from commonroad_geometric.learning.geometric.project.hydra_geometric_config import GeometricProjectConfig
from commonroad_geometric.learning.geometric.training.callbacks.callback_computer_container_service import CallbackComputerContainerService, CallbackComputersContainer
from commonroad_geometric.learning.geometric.training.callbacks.implementations.early_stopping_callback import EarlyStoppingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.epoch_checkpoint_callback import EpochCheckpointCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.export_latest_model_callback import ExportLatestModelCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.gradient_clipping_callback import GradientClippingCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.log_wandb_callback import LogWandbCallback
from commonroad_geometric.learning.geometric.training.callbacks.implementations.watch_model_callback import WatchWandbCallback
from commonroad_geometric.learning.geometric.training.experiment import GeometricExperiment, GeometricExperimentConfig
from commonroad_geometric.learning.geometric.base_geometric import MODEL_FILENAME, MODEL_FILETYPE, MODEL_FILE, OPTIMIZER_FILE, STATE_DICT_FILE, BaseGeometric
from commonroad_geometric.learning.geometric.training.geometric_trainer import GeometricTrainer
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService


logger = logging.getLogger(__name__)


class GeometricProjectDirectoryStructure:
    def __init__(
        self,
        cfg: GeometricProjectConfig
    ) -> None:

        project_dir = Path(cfg.project_dir).resolve()
        train_scenario_dir = Path(cfg.dataset.train_scenario_dir).resolve()
        test_scenario_dir = Path(cfg.dataset.test_scenario_dir).resolve()
        val_scenario_dir = Path(cfg.dataset.val_scenario_dir).resolve()
        self.project_dir = project_dir
        self.train_scenario_dir = train_scenario_dir
        self.test_scenario_dir = test_scenario_dir
        self.val_scenario_dir = val_scenario_dir
        self.model_dir = project_dir.joinpath( 'model')
        self.latest_model_dir = self.model_dir.joinpath('latest')
        self.video_dir = project_dir.joinpath( 'videos')
        self.train_dataset_dir = project_dir.joinpath( 'dataset/train')
        self.test_dataset_dir = project_dir.joinpath( 'dataset/test')
        
        self.latest_model_dir.mkdir(exist_ok=True, parents=True)
        self.video_dir.mkdir(exist_ok=True, parents=True)
        self.train_dataset_dir.mkdir(exist_ok=True, parents=True)
        self.test_dataset_dir.mkdir(exist_ok=True, parents=True)

    def get_model_checkpoint_dir(self, checkpoint: Optional[str]) -> str:
        if checkpoint is None:
            checkpoint_dir = self.latest_model_dir
        else:
            if os.path.isdir(checkpoint):
                checkpoint_dir = checkpoint
            else:
                checkpoint_dir = os.path.dirname(search_file(
                    self.model_dir,
                    search_term=checkpoint,
                    file_name=MODEL_FILENAME,
                    file_type=MODEL_FILETYPE
                ))
        return checkpoint_dir

class BaseGeometricProject(BaseProject):
    """
    Abstract base class for facilitating geometric deep learning projects.
    Requires experiment configuration and model implementation to be overridden.
    See the tutorials for example usage.
    """

    def __init__(
        self, 
        cfg: GeometricProjectConfig
    ) -> None:
        self.cfg = cfg

        setup_logging(
            level=self.cfg.logging_level
        )

        self.model_cfg = Config(self.cfg.model)
        self.experiment_cfg = Config(self.cfg.experiment)
        self.run_modes: Dict[str, Callable[[], None]]

        self.dir_structure = GeometricProjectDirectoryStructure(self.cfg)
        set_global_seed(self.cfg.seed) 

        self.experiment = GeometricExperiment(self.configure_experiment(self.experiment_cfg))
        self.model_cls = self.configure_model(self.model_cfg, self.experiment)

        if self.cfg.device == 'auto':
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)
        
        self.project_name = f"GeometricProject_{self.model_class_name}"

        super().__init__()

    @abstractmethod
    def configure_experiment(self, cfg: Config) -> GeometricExperimentConfig:
        ...

    @abstractmethod
    def configure_model(self, cfg: Config, experiment: GeometricExperiment) -> Type[BaseGeometric]:
        ...

    def configure_training_callbacks(
        self, 
        wandb_service: WandbService
    ) -> CallbackComputersContainer:
        callbacks_computers = CallbackComputersContainer(
            training_step_callbacks=CallbackComputerContainerService([
                ExportLatestModelCallback(
                    directory=self.dir_structure.latest_model_dir,
                    save_frequency=self.cfg.training.checkpoint_frequency if self.cfg.training.validate_inner else 1
                ),
                LogWandbCallback(wandb_service=wandb_service),
                GradientClippingCallback(self.cfg.training.gradient_clipping_threshold)
                # DebugTrainBackwardGradientsCallback(frequency=200)
            ]),
            validation_step_callbacks=CallbackComputerContainerService([
                #LogInfoCallback()
            ]),
            logging_callbacks=CallbackComputerContainerService([LogWandbCallback(wandb_service=wandb_service)]),
            initialize_training_callbacks=CallbackComputerContainerService([WatchWandbCallback(
                wandb_service=wandb_service,
                log_freq=self.cfg.training.log_freq,
                log_gradients=not self.cfg.warmstart
            )]),
            checkpoint_callbacks=CallbackComputerContainerService([EpochCheckpointCallback(
                directory=self.dir_structure.model_dir
            )]),
            early_stopping_callbacks=CallbackComputerContainerService([EarlyStoppingCallback(
                after_epochs=self.cfg.training.early_stopping
            )])
        )
        return callbacks_computers

    @property
    def model_class_name(self) -> str:
        try:
            model_cls_name = self.model_cls.__name__
        except AttributeError:
            if isinstance(self.model_cls, partial):
                model_cls_name = self.model_cls.func.__name__
            else:
                raise NotImplementedError(type(self.model_cls))
        return model_cls_name
        
    def get_train_dataset(
        self, 
        force_dataset_overwrite: bool = False,
        collect_missing_samples: bool = True
    ) -> CommonRoadDataset:
        return self.get_dataset(
            scenario_dir=self.dir_structure.train_scenario_dir,
            dataset_dir=self.dir_structure.train_dataset_dir,
            force_dataset_overwrite=force_dataset_overwrite,
            collect_missing_samples=collect_missing_samples
        )
        
    def get_test_dataset(
        self, 
        force_dataset_overwrite: bool = False,
        collect_missing_samples: bool = True
    ) -> CommonRoadDataset:
        return self.get_dataset(
            scenario_dir=self.dir_structure.test_scenario_dir,
            dataset_dir=self.dir_structure.test_dataset_dir,
            force_dataset_overwrite=force_dataset_overwrite,
            collect_missing_samples=collect_missing_samples
        )

    def get_dataset(
        self, 
        scenario_dir: Path,
        dataset_dir: Path,
        force_dataset_overwrite: bool = False,
        collect_missing_samples: bool = True
    ) -> CommonRoadDataset:
        dataset_cfg = self.cfg.dataset

        return self.experiment.get_dataset(
            scenario_dir=scenario_dir,
            dataset_dir=dataset_dir,
            overwrite=force_dataset_overwrite,
            pre_transform_workers=dataset_cfg.pre_transform_workers,
            max_scenarios=dataset_cfg.max_scenarios,
            cache_data=dataset_cfg.cache_data,
            max_samples_per_scenario=dataset_cfg.max_samples_per_scenario,
            collect_missing_samples=collect_missing_samples
        )

    @register_run_command
    def render(self) -> None:
        model, _ = self._get_model_and_optimizer_state(force_build=True)
        render_model(
            model=model,
            experiment=self.experiment,
            scenario_path=self.dir_structure.val_scenario_dir,
            disable_postprocessing=self.cfg.disable_postprocessing_inference,
            disable_inference=False,
            disable_compute_loss=True,
            loop_scenarios=True,
        )

    @register_run_command
    def render_latest(self) -> None:
        model_path = self._get_latest_model_path()
        render_model(
            model=model_path,
            experiment=self.experiment,
            scenario_path=self.dir_structure.val_scenario_dir,
            disable_postprocessing=self.cfg.disable_postprocessing_inference,
            disable_inference=False,
            disable_compute_loss=True,
            loop_scenarios=True
        )

    @register_run_command
    def record_latest(self) -> None:
        model_path = self._get_latest_model_path()
        render_model(
            model=model_path,
            experiment=self.experiment,
            scenario_path=self.dir_structure.val_scenario_dir,
            disable_postprocessing=self.cfg.disable_postprocessing_inference,
            disable_inference=False,
            disable_compute_loss=True,
            video_dir=self.dir_structure.video_dir,
            loop_scenarios=False,
            screenshot_freq=10
        )

    @register_run_command
    def collect_train(self) -> None:
        shutil.rmtree(self.dir_structure.train_dataset_dir, ignore_errors=True)
        self.get_train_dataset(force_dataset_overwrite=True)

    @register_run_command
    def collect_test(self) -> None:
        shutil.rmtree(self.dir_structure.test_dataset_dir, ignore_errors=True)
        self.get_test_dataset(force_dataset_overwrite=True)

    @register_run_command
    def transform_train(self) -> None:
        dataset = self.get_train_dataset(force_dataset_overwrite=False)
        self.experiment.transform_dataset(dataset)

    @register_run_command
    def transform_test(self) -> None:
        dataset = self.get_test_dataset(force_dataset_overwrite=False)
        self.experiment.transform_dataset(dataset)

    @register_run_command
    def train(self) -> None:
        model, optimizer_state = self._get_model_and_optimizer_state()

        model.train(True)
        dataset = self.get_train_dataset(force_dataset_overwrite=False)

        wandb_service = WandbService(
            disable=self.cfg.no_wandb,
            project_name=self.project_name
        )
        wandb_metadata = dict(
            model_cls=self.model_class_name,
            model=self.cfg.model,
            experiment=self.cfg.experiment,
            training=self.cfg.training
        )
        wandb_kwargs = {}

        experiment_name = f"{self.model_class_name}-{get_timestamp_filename()}"

        trainer = GeometricTrainer(
            cfg=self.cfg.training,
            device=self._device
        )
        if trainer.multi_gpu():
            wandb_kwargs["group"] = f'{type(model).__name__}-DDP'

        wandb_experiment_args = {
            'name': experiment_name,
            'metadata': wandb_metadata,
            'include_timestamp': False,
            **wandb_kwargs
        }
        if not trainer.multi_gpu():
            experiment_name = wandb_service.start_experiment(**wandb_experiment_args)
        
        # TODO: https://gitlab.lrz.de/cps/commonroad-geometric/-/issues/229
        if isinstance(self.experiment.config.extractor_factory, TemporalTrafficExtractorFactory):
            def noop_collate_fn(data_list: List[T_Data]) -> T_Data:
                assert len(data_list) == 1
                return data_list[0]
            collate_fn = noop_collate_fn
        else:
            collate_fn = None

        trainer.launch_trainer(
            model_dir=os.path.join(self.dir_structure.model_dir, experiment_name),
            experiment=self.experiment,
            dataset=dataset,
            model=model,
            wandb_service=wandb_service,
            wandb_experiment_args=wandb_experiment_args,
            optimizer_state=optimizer_state,
            collate_fn=collate_fn,
            callbacks_computers=self.configure_training_callbacks(wandb_service),
            render_scenario_path=self.dir_structure.val_scenario_dir
        )

    def _get_latest_model_path(self) -> Path:
        model_path = get_most_recent_file(
            list_files(
                self.dir_structure.project_dir, 
                file_name=MODEL_FILENAME, 
                file_type=MODEL_FILETYPE, 
                sub_directories=True
            )
        )
        return Path(model_path)

    def _get_model_and_optimizer_state(
        self,
        force_build: bool = False
    ) -> Tuple[BaseGeometric, Optional[Dict[str, Any]]]:
        checkpoint_dir = self.dir_structure.get_model_checkpoint_dir(self.cfg.checkpoint)
        
        if self.cfg.warmstart:
            model = load_dill(checkpoint_dir / MODEL_FILE)
            optimizer_state = torch.load(
                checkpoint_dir / OPTIMIZER_FILE,
                map_location=self._device
            )
        else:
            model = self.model_cls(self.model_cfg)
            if force_build:
                dataset = self.get_train_dataset(
                    force_dataset_overwrite=False,
                    collect_missing_samples=False
                )
                model.build(data=dataset[0])
            optimizer_state = None
        model.eval()
        return model, optimizer_state