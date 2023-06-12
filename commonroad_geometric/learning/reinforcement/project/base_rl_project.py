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
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService

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

    def __init__(
        self, 
        cfg: RLProjectConfig
    ) -> None:
        self.cfg = cfg

        set_global_seed(self.cfg.seed) 

        self.experiment = RLExperiment(self.configure_experiment(Config(self.cfg.experiment)))
        self.model_cfg = self.configure_model(Config(self.cfg.model), self.experiment)

        if self.cfg.device == 'auto':
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)

        self.project_name = f"RLProject_{self.experiment.config.control_space_cls.__name__}"

    @register_run_command
    def train(self) -> None:
        checkpoint = self._get_model_checkpoint(self.cfg.checkpoint)

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
                warmstart=self.cfg.warmstart
            )
        )
        trainer.train(device=self._device)

    @register_run_command
    def record(self) -> None:
        # TODO: cleanup
        trainer = self._init_trainer()
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir)
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
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir)
        trainer.enjoy()

    @register_run_command
    def play(self) -> None:
        # TODO: cleanup
        trainer = self._init_trainer()
        trainer.init_agent(device='cpu', scenario_dir=self.cfg.scenario_dir)
        trainer.play(predict_agent=False)

    @register_run_command
    def benchmark(self) -> None:
        # TODO
        raise NotImplementedError()

        benchmark_dir = Config.BENCHMARKING_DIR
        os.makedirs(benchmark_dir, exist_ok=True)

        model_dir = model_dir if model_dir is not None else Config.DEFAULT_OUTPUT_DIR

        model_paths = list_files(
            model_dir,
            file_name='best_model',
            file_type='zip',
            join_paths=True,
            sub_directories=True
        )

        def setup_wandb_service(
            project_name: str,
            experiment_name: str
        ) -> WandbService:
            wandb_service = WandbService(disable=args.no_wandb, project_name=project_name)
            experiment_name = wandb_service.start_experiment(
                name=experiment_name,
                include_timestamp=False
            )
            return wandb_service

        for model_path in model_paths:
            experiment_path = search_file(os.path.dirname(os.path.dirname(model_path)), 'experiment_config')
            experiment = RLExperiment.load(experiment_path)
            env = experiment.make_env(
                scenario_dir,
                seed=seed,
                renderer_options=RenderConfig.RENDERER_OPTION_LIST[0],
                render_debug_overlays=False
            )

            model_benchmark_dir = os.path.join(benchmark_dir, model_path.name)
            os.makedirs(model_benchmark_dir, exist_ok=True)

            agent = model_cls.load(
                path=model_path,
                env=env,
                device=device,
                seed=seed
            )

            print("\n----------")
            print(f"Benchmarking {model_path}")
            wandb_service = setup_wandb_service(
                project_name=f"benchmark-graph-rl",
                experiment_name=message.replace(' ', '-') + '-' + model_path.name.replace(' ', '-')
            )

            aggregate_statistics: Dict[str, List[float]] = defaultdict(list)

            for obs, reward, done, info in render_agent(
                agent=agent,
                experiment=experiment,
                env=env,
                scenario_dir=scenario_dir,
                video_folder=model_benchmark_dir,
                total_timesteps=3000,
                deterministic=True,
                seed=seed
            ):
                if done:
                    episode_summary = on_episode_end(env, info)
                    print(episode_summary)
                    for k, v in episode_summary.items():
                        aggregate_statistics[k].append(v)
            
            log_dict: Dict[str, float] = {}
            for k, v in aggregate_statistics.items():
                values = np.array(v)
                mean_value = values.mean()
                max_value = values.max()
                min_value = values.min()
                std_value = values.std()
                log_dict[f"{k}_mean"] = mean_value
                log_dict[f"{k}_max"] = max_value
                log_dict[f"{k}_min"] = min_value
                log_dict[f"{k}_std"] = std_value

            wandb_service.log(log_dict)
            wandb_service.finish_experiment()

    def _init_trainer(self) -> RLTrainer:
        checkpoint = self._get_model_checkpoint(self.cfg.checkpoint)
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
                warmstart=self.cfg.warmstart
            )
        )
        return trainer

    def _get_model_checkpoint(self, checkpoint: Optional[str]) -> Optional[str]:
        if checkpoint is None:
            files = list_files(
                os.path.dirname(self.cfg.project_dir),
                file_type='zip',
                sub_directories=True
            )
            if files:
                checkpoint = get_most_recent_file(files)
        return checkpoint