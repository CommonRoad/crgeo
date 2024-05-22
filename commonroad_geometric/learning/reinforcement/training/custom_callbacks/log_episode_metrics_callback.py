import copy
import logging
import warnings
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from commonroad_geometric.learning.reinforcement.training.custom_callbacks.common import _ArrayBufferMetrics

logger = logging.getLogger(__name__)


class LogEpisodeMetricsCallback(BaseCallback):
    """
    Records metrics at the end of each training episode.
    """

    BUFFER_METRICS_ON_EPISODE_END = ['actions', 'log_probs', 'rewards', 'values']
    BUFFER_METRICS_ON_ROLLOUT_END = ['advantages', 'values']

    def __init__(self, verbose: int = 0):
        super(LogEpisodeMetricsCallback, self).__init__(verbose=verbose)
        self.n_episodes: int = 0
        self._info_buffer: Dict[int, List[Dict[str, Any]]]
        self.n_rollouts: int = 0
        self._disabled = False
        self._disabled = False

    def _on_training_start(self) -> None:
        self._termination_reasons = self.training_env.get_attr('termination_reasons')[0]

    def _on_rollout_start(self) -> None:
        self._info_buffer = {}

    def _on_step(self) -> bool:
        if self._disabled:
            return True
        assert self.logger is not None

        n_steps = self.locals.get('n_steps')
        if n_steps is None:
            warnings.warn("n_steps is None, disabling ")
            self._disabled = True
            return True
        rollout_buffer = self.locals.get('rollout_buffer')
        infos = copy.deepcopy(self.locals.get('infos'))  # TODO: Necessary?

        last_info = self._info_buffer.get(n_steps - 1, None)
        last_done_array = np.array([info['done'] for info in last_info]) if last_info is not None else None

        n_episodes_done_step = np.sum(last_done_array).item() if last_done_array is not None else 0
        if n_episodes_done_step > 0:
            try:
                for attr in LogEpisodeMetricsCallback.BUFFER_METRICS_ON_EPISODE_END:
                    buffer_metrics = LogEpisodeMetricsCallback._analyze_buffer_array_masked(
                        buffer=getattr(rollout_buffer, attr),
                        n_steps=n_steps,
                        done_array=last_done_array
                    )
                    metric_dict = asdict(buffer_metrics)
                    for metric_name, value in metric_dict.items():
                        if isinstance(value, np.ndarray):
                            for idx, idx_value in enumerate(value):
                                self.logger.record(f"train/{attr}_ep_{metric_name}_idx_{idx}", idx_value)
                        else:
                            self.logger.record(f"train/{attr}_ep_{metric_name}", value)
            except Exception as e:
                logger.error(e, exc_info=True)

            termination_criteria_flags = dict.fromkeys(self._termination_reasons, False)
            done_indices = np.where(last_done_array)[0]
            for env_index in done_indices:
                env_info = last_info[env_index]
                termination_reason = env_info.get('termination_reason')
                termination_criteria_flags[termination_reason] = True

                env_reward_component_info = env_info['reward_component_episode_info']
                assert env_reward_component_info is not None
                for reward_component, component_info in env_reward_component_info.items():
                    for component_metric, component_value in component_info.items():
                        self.logger.record(f"train/reward_{reward_component}_ep_{component_metric}", component_value)

                vehicle_aggregate_stats = env_info['vehicle_aggregate_stats']
                assert vehicle_aggregate_stats is not None
                for state, state_info in vehicle_aggregate_stats.items():
                    for state_metric, state_value in state_info.items():
                        self.logger.record(f"train/vehicle_{state}_ep_{state_metric}", state_value)

                num_obstacles = env_info.get('total_num_obstacles')
                self.logger.record("train/ep_num_obstacles", num_obstacles)

                cumulative_reward = env_info.get('cumulative_reward')
                self.logger.record("train/ep_cumulative_reward", cumulative_reward)

                next_reset_ready = env_info.get('next_reset_ready')
                self.logger.record("train/next_reset_ready", float(next_reset_ready))

                episode_length = env_info.get('time_step')
                self.logger.record("train/ep_length", episode_length)

            for termination_criteria in self._termination_reasons:
                self.logger.record(
                    f"train/termination_{termination_criteria}",
                    termination_criteria_flags[termination_criteria])

            self.n_episodes += n_episodes_done_step

            self.logger.record("info/n_episodes", self.n_episodes)
            self.logger.dump(step=self.num_timesteps)

        self._info_buffer[n_steps] = infos
        self._info_buffer = {step: v for step, v in self._info_buffer.items() if step >= n_steps - 5}

        return True

    def _on_training_end(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.locals.get('rollout_buffer')
        if rollout_buffer is not None:
            for attr in LogEpisodeMetricsCallback.BUFFER_METRICS_ON_ROLLOUT_END:
                buffer_metrics = LogEpisodeMetricsCallback._analyze_buffer_array(
                    buffer=getattr(rollout_buffer, attr)
                )
                metric_dict = asdict(buffer_metrics)
                for metric_name, value in metric_dict.items():
                    if isinstance(value, np.ndarray):
                        for idx, idx_value in enumerate(value):
                            self.logger.record(f"train/{attr}_ep_{metric_name}_idx_{idx}", idx_value)
                    else:
                        self.logger.record(f"train/{attr}_ep_{metric_name}", value)
        self.logger.dump(step=self.num_timesteps)
        self.n_rollouts += 1

    @staticmethod
    def _analyze_buffer_array_masked(buffer: np.ndarray, n_steps: int, done_array: np.ndarray) -> _ArrayBufferMetrics:
        masked_buffer = buffer[:n_steps, done_array, ...]
        metrics = LogEpisodeMetricsCallback._analyze_buffer_array(masked_buffer)
        return metrics

    @staticmethod
    def _analyze_buffer_array(buffer: np.ndarray) -> _ArrayBufferMetrics:
        metrics = _ArrayBufferMetrics(
            max=np.max(buffer, axis=(0, 1)),
            min=np.min(buffer, axis=(0, 1)),
            std=np.std(buffer, axis=(0, 1)),
            mean=np.mean(buffer, axis=(0, 1)),
            absmean=np.mean(np.abs(buffer), axis=(0, 1))
        )

        return metrics
