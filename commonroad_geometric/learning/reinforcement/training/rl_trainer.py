from __future__ import annotations

import logging
import os
import os.path
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Type, Union

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from wandb.integration.sb3 import WandbCallback

from commonroad_geometric.common.logging import stdout
from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment
from commonroad_geometric.learning.reinforcement.training.custom_callbacks import EvalCallback, LogEpisodeMetricsCallback, LogPolicyMetricsCallback, RecordVideoSubprocCallback
from commonroad_geometric.learning.reinforcement.training.types import T_GymEnvironment
from commonroad_geometric.learning.reinforcement.training.utils.performance_metrics import on_episode_end
from commonroad_geometric.learning.reinforcement.training.utils.render_agent import render_agent
from commonroad_geometric.learning.training.wandb_service import WandbService
from commonroad_geometric.simulation.ego_simulation.control_space.keyboard_input import UserAdvanceScenarioInterrupt, UserQuitInterrupt, UserResetInterrupt, get_keyboard_action

logger = logging.getLogger(__name__)


@dataclass
class RLModelConfig:
    agent_cls: Type[BaseAlgorithm]
    agent_kwargs: dict = field(default_factory=dict)

@dataclass
class RLTrainerConfig:
    checkpoint_freq: int = 10000
    eval_freq: Optional[int] = 10
    gradient_save_freq: int = 1
    log_freq: int = 1
    n_envs: int = 1
    n_eval_episodes: int = 10
    normalize_observations: bool = False
    normalize_rewards: bool = False
    normalize_rewards_threshold: float = 10.0
    record_backoff: float = 1.0
    record_freq: Optional[int] = 10000
    total_timesteps: int = 1_000_000
    verbose: int = 0
    video_length: int = 200
    wandb_logging: bool = True


@dataclass
class RLTrainerParams:
    experiment: RLExperiment
    project_name: str
    scenario_dir: str
    output_dir: Path
    checkpoint: Optional[str]
    warmstart: bool
    seed: int
    model_cfg: RLModelConfig
    train_cfg: RLTrainerConfig


class RLTrainer:
    def __init__(
        self,
        params: RLTrainerParams
    ) -> None:
        """Trainer for reinforcement learning agents."""
        self.params = params
        self.train_cfg = params.train_cfg
        self.model_cfg = params.model_cfg
        self._agent: Optional[BaseAlgorithm] = None

    @property
    def agent(self) -> Optional[BaseAlgorithm]:
        return self._agent

    def play(
        self,
        predict_agent: bool = True,
    ) -> None:
        env = self.params.experiment.make_env(scenario_dir=self.params.scenario_dir)
        experiment_name = self.params.experiment.create_name()
        if predict_agent and self._agent is None:
            self.init_agent(
                device='cpu',
                seed=self.params.seed,
                scenario_dir=self.params.scenario_dir,
                output_dir=os.path.join(self.params.output_dir, 'play', experiment_name) if self.params.output_dir is not None else None
            )
        obs = env.reset()
        logger.info("Playing environment as game with keyboard input. Press R to reset and Q to quit.")
        
        total_reward: float = 0.0
        while 1:
            if predict_agent:
                agent_action, _ = self.agent.predict(obs)
            else:
                agent_action = None
            try:
                action = get_keyboard_action(renderer=env.get_attr('renderers')[0][0])
            except UserResetInterrupt:
                env.env_method('respawn')
                action = np.array([0.0, 0.0], dtype=np.float32)
            except UserAdvanceScenarioInterrupt:
                env.reset()
                total_reward = 0.0
                action = np.array([0.0, 0.0], dtype=np.float32)
            except UserQuitInterrupt:
                logger.info("Quit game")
                return
            obs, reward, done, info = env.step([action])
            total_reward += reward.item()
            msg = f"reward: {reward.item():.3f} ({total_reward:.3f}), low: {info[0]['lowest_reward_computer']} ({info[0]['lowest_reward']:.3f}), high: {info[0]['highest_reward_computer']} ({info[0]['highest_reward']:.3f}), t: {info[0]['time_step']}"
            if predict_agent:
                msg += f", agent: {numpy_prettify(agent_action)}"
            stdout(msg)
            if not done:
                env.render()
            else:
                total_reward = 0.0

    def train(
        self,
        device: torch.device
    ) -> None:
        n_envs = self.train_cfg.n_envs
        if n_envs < 0:
            import multiprocessing
            n_envs = multiprocessing.cpu_count()

        experiment_name = self.params.experiment.create_name()

        if self.train_cfg.wandb_logging:
            wandb_service = WandbService(
                project_name=self.params.project_name
            )
            experiment_name = wandb_service.start_experiment(
                name=experiment_name,
                metadata=dict(
                    params=self.params,
                    config=self.train_cfg
                ),
                include_timestamp=False,
                include_random_name=True,
                update_terminal_title=True,
                sync_tensorboard=True
            )

        output_dir = os.path.join(self.params.output_dir, experiment_name) if self.params.output_dir is not None else None

        logger.debug(f"Creating {n_envs} training environment(s) with seed {self.params.seed}")
        train_env = self.params.experiment.make_env(self.params.scenario_dir, n_envs, seed=self.params.seed)

        if self.train_cfg.normalize_rewards or self.train_cfg.normalize_observations:
            train_env = VecNormalize(
                train_env,
                norm_obs=self.train_cfg.normalize_observations,
                norm_reward=self.train_cfg.normalize_rewards,
                clip_reward=self.train_cfg.normalize_rewards_threshold,
            )
            if not self.train_cfg.normalize_observations:
                # monkeypatch to disable unnecessary deepcopy call from SB3 implementation
                def disable_normalize_obs(obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
                    return obs
                train_env.normalize_obs = disable_normalize_obs

        self.params.experiment.config.env_options = train_env.get_attr('options')[0] # hack to standardize observation space
        if output_dir is not None:
            experiment_file_path = self.params.experiment.save(output_dir)

        env_seed = self.params.seed + n_envs
        logger.debug(f"Creating eval environment with seed {env_seed}")
        eval_env = self.params.experiment.make_env(self.params.scenario_dir, n_envs, seed=env_seed)

        try:
            self.init_agent(
                env=train_env,
                device=device,
                seed=self.params.seed,
                output_dir=output_dir,
                checkpoint=self.params.checkpoint if self.params.warmstart else None
            )
        except ValueError:
            logger.warn("Failed to warmstart agent - initializing new")
            self.init_agent(
                env=train_env,
                device=device,
                seed=self.params.seed,
                output_dir=output_dir,
                checkpoint=None
            )

        callbacks = [
            LogEpisodeMetricsCallback(),
            LogPolicyMetricsCallback()
        ]
        if output_dir is not None:
            callbacks.append(CheckpointCallback(
                save_freq=max(self.train_cfg.checkpoint_freq // n_envs, 1),
                save_path=os.path.join(output_dir, 'checkpoints'),
                name_prefix=experiment_name,
                verbose=self.train_cfg.verbose
            ))
            if self.train_cfg.eval_freq is not None:
                callbacks.append(
                    EvalCallback(
                        eval_env=eval_env,
                        n_eval_episodes=self.train_cfg.n_eval_episodes,
                        eval_freq=max(self.train_cfg.eval_freq // n_envs, 1),
                        log_path=os.path.join(output_dir, 'evaluations'),
                        best_model_save_path=os.path.join(output_dir, 'best'),
                        verbose=self.train_cfg.verbose
                    )
                )

            if self.train_cfg.record_freq is not None:
                callbacks.append(
                    RecordVideoSubprocCallback(
                        scenario_dir=self.params.scenario_dir,
                        experiment_file=experiment_file_path,
                        record_freq=self.train_cfg.record_freq,
                        record_backoff=self.train_cfg.record_backoff,
                        video_folder=os.path.join(output_dir, 'videos'),
                        video_length=self.train_cfg.video_length,
                        device='cpu'
                    )
                )

        if self.train_cfg.wandb_logging and wandb_service.success:
            callbacks.append(WandbCallback(
                gradient_save_freq=self.train_cfg.gradient_save_freq,
                model_save_path=os.path.join(output_dir, 'wandb-models') if output_dir is not None else None,
                verbose=2,
            ))
        callbacks = CallbackList(callbacks)

        try:
            self._agent.learn(
                total_timesteps=self.train_cfg.total_timesteps,
                callback=callbacks,
                log_interval=self.train_cfg.log_freq,
                tb_log_name=experiment_name
            )
        except KeyboardInterrupt as e:
            # Exiting gracefully
            train_env.close()
            eval_env.close()
            raise e
        finally:
            if self.train_cfg.wandb_logging:
                wandb_service.finish_experiment()
            logger.info('Training finished')

    def enjoy(
        self,
        total_timesteps: Optional[int] = None,
        deterministic: Optional[Union[Callable[[int], bool], bool]] = None
    ) -> Iterable[Dict[str, float]]:
    
        if deterministic is None:
            deterministic = lambda i: int(i // 50) % 2 == 0

        for obs, reward, done, info in render_agent(
            agent=self._agent,
            env=self._env,
            total_timesteps=total_timesteps,
            verbose=2,
            deterministic=deterministic
        ):
            if done:
                episode_summary = on_episode_end(self._env, info)
                yield episode_summary

    def record(
        self,
        scenario_dir: str,
        video_folder: str,
        video_length: Optional[int] = None,
        deterministic: bool = True,
        seed: int = 0,
        n_videos: int = 1
    ) -> Iterable[Dict[str, float]]: # TODO redundant
        os.makedirs(video_folder, exist_ok=True)

        env = self.params.experiment.make_env(scenario_dir, seed=seed)
        for i in range(n_videos):
            list(render_agent(
                agent=self._agent,
                env=env,
                experiment=self.params.experiment,
                scenario_dir=scenario_dir,
                video_folder=video_folder,
                total_timesteps=video_length,
                deterministic=deterministic,
                seed=seed,
                break_on_done=True
            ))

    def init_agent(
        self,
        *,
        device: Optional[torch.device] = None,
        env: Optional[Optional[T_GymEnvironment]] = None,
        seed: int = 0,
        output_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        scenario_dir: Optional[str] = None
    ) -> None:
        if device is None:
            device = self.train_cfg

        if env is None:
            assert scenario_dir is not None
            env = self.params.experiment.make_env(
                scenario_dir=scenario_dir
            )
        self._env = env
        tensorboard_log = os.path.join(output_dir, 'tensorboard') if output_dir is not None else None
        if checkpoint is None:
            self._agent = self.model_cfg.agent_cls(
                env=env,
                device=device,
                tensorboard_log=tensorboard_log,
                seed=seed,
                **self.model_cfg.agent_kwargs
            )

        else:
            self._agent = self.model_cfg.agent_cls.load(
                path=checkpoint,
                env=env,
                device=device,
                tensorboard_log=tensorboard_log,
                seed=seed,
                **self.model_cfg.agent_kwargs
            )

        logger.info(f"Loaded {self.model_cfg.agent_cls.__name__} agent from {checkpoint}")
