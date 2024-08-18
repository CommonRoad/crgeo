import os
import warnings
from typing import Any, Dict, Optional, Union

import gymnasium
import logging
import numpy as np
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from commonroad_geometric.learning.reinforcement.training.rl_trainer import *

def get_depopulate_preprocessors(env):
    preprocessors = [s.preprocessor for s in env.get_attr("scenario_iterator")]
    depopulate_scenario_preprocessors = [p if p.__class__.__name__ == "DepopulateScenarioPreprocessor" else next(c for c in p.child_preprocessors if c.__class__.__name__ == 'DepopulateScenarioPreprocessor') for p in preprocessors]
    return depopulate_scenario_preprocessors

class TrafficDensityCurriculumCallback(BaseCallback):
    """
    Callback for changing the training and evaluation environment and agent properties based on curriculum stages.
    """

    def __init__(self, increase_rate: float):
        self.increase_rate = increase_rate
        self.stage = 1
        self.n_rollouts = 0
        # for episode based curriculum
        self.max_episodes = 0
        self.n_episodes = 0
        super().__init__()

    def _on_rollout_end(self) -> None:
        """
        Method called after policy rollout.
        
        Setting all relevant parameters according to new stage specification in both train and evaluation environment. 
        """
        try:
            self.eval_callback = next(c for c in self.locals['callback'].callbacks if c.__class__.__name__ == 'EvalCallback')
        except:
            self.eval_callback = None

        depopulate_input = min(1.0, self.increase_rate*self.n_episodes)
        for depopulator in get_depopulate_preprocessors(self.training_env.unwrapped):
            depopulator.depopulator = depopulate_input
        get_depopulate_preprocessors(self.eval_callback.eval_env.unwrapped)[0].depopulator = depopulate_input

        print(f"Updated depopulator value to {depopulate_input:.3f} (total episodes: {self.n_episodes})")

        return

    def _on_step(self) -> bool:
        """
        Method called after each agent step (action). Updates the count of finished episodes.

        If the simulation variable of train or eval env is stored for use in self it should be retreived here fresh after a 
        detected episode change, as every two episodes new simulation instances get created and used in train/eval env.
        """
        # needs to be executed on every step, otherwise finished episodes don't get tracked
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        return True
