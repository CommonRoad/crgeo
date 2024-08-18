from __future__ import annotations

import logging
import os
from abc import abstractmethod
from typing import Dict, Optional

import torch

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.utils.filesystem import get_most_recent_file, list_files, search_file
from commonroad_geometric.common.utils.seeding import set_global_seed
from commonroad_geometric.learning.base_project import BaseProject, register_run_command
from commonroad_geometric.learning.reinforcement import RLTrainer
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.project.hydra_rl_config import RLProjectConfig
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig, RLTrainerParams
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

logger = logging.getLogger(__name__)


class BaseRLProject(BaseProject):
    """
    Abstract base class for facilitating geometric reinforcement learning projects.
    See the tutorials for example usage.
    """

    @abstractmethod
    def configure_experiment(self, cfg: Config) -> RLExperimentConfig:
        ...

    @abstractmethod
    def configure_model(self, cfg: Config, experiment: RLExperiment) -> RLModelConfig:
        ...

    def configure_custom_callbacks(self, cfg: Config) -> list[BaseCallback]:
        return []

    def __init__(
        self, 
        cfg: RLProjectConfig
    ) -> None:
        self.cfg = cfg

        set_global_seed(self.cfg.seed) 

        self.experiment = RLExperiment(self.configure_experiment(Config(self.cfg.experiment)))
        self.model_cfg = self.configure_model(Config(self.cfg.model), self.experiment)
        self.custom_callbacks = self.configure_custom_callbacks(Config(self.cfg.experiment))

        if self.cfg.device == 'auto':
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)

        self.project_name = f"RLProject_{type(self).__name__}"

    @register_run_command
    def train(self) -> None:
        checkpoint = self._get_agent_checkpoint(self.cfg.checkpoint)

        trainer = RLTrainer(
            params=RLTrainerParams(
                checkpoint=checkpoint,
                experiment=self.experiment,
                model_cfg=self.model_cfg,
                output_dir=self.cfg.project_dir,
                project_name=self.project_name,
                scenario_dir=self.cfg.scenario_dir,
                seed=self.cfg.seed,
                train_cfg=self.cfg.training,
                project_cfg=self.cfg,
                warmstart=self.cfg.warmstart,
                custom_callbacks=self.custom_callbacks
            )
        )
        trainer.train(
            device=self._device
        )

    @register_run_command
    def record(self) -> None:
        # TODO: cleanup
        trainer = self._init_trainer()
        trainer.init_agent(
            device='cpu', 
            scenario_dir=self.cfg.scenario_dir,
        )
        trainer.record(
            video_folder=self.cfg.project_dir / 'videos/',
            video_length=None,
            scenario_dir=self.cfg.scenario_dir,
            n_videos=1
        )

    @register_run_command
    def enjoy(self) -> None:
        # TODO: cleanup
        trainer = self._init_trainer()
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir, checkpoint=trainer.params.checkpoint)
        for episode_summary in trainer.enjoy():
            print(episode_summary)

    @register_run_command
    def play(self) -> None:
        trainer = self._init_trainer()
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir)
        trainer.play(predict_agent=False)

    @register_run_command
    def play_agent(self) -> None:
        trainer = self._init_trainer()
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir, checkpoint=trainer.params.checkpoint)
        trainer.play(predict_agent=True)

    def _init_trainer(self) -> RLTrainer:
        checkpoint = self._get_agent_checkpoint(self.cfg.checkpoint)
        trainer = RLTrainer(
            params=RLTrainerParams(
                checkpoint=checkpoint,
                experiment=self.experiment,
                model_cfg=self.model_cfg,
                output_dir=self.cfg.project_dir,
                project_name=self.project_name,
                scenario_dir=self.cfg.scenario_dir,
                seed=self.cfg.seed,
                train_cfg=self.cfg.training,
                warmstart=self.cfg.warmstart,
                project_cfg=self.cfg,
                custom_callbacks=self.custom_callbacks
            )
        )
        return trainer

    def _get_agent_checkpoint(self, checkpoint: Optional[str]) -> Optional[str]:
        if checkpoint is None:
            files = list_files(
                self.cfg.project_dir.parent,
                file_type='zip',
                sub_directories=True
            )
            if files:
                checkpoint = get_most_recent_file(files)
        return checkpoint