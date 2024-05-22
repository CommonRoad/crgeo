from typing import Union

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from commonroad_geometric.learning.reinforcement.commonroad_gym_env import CommonRoadGymEnv

T_GymEnvironment = Union[DummyVecEnv, SubprocVecEnv, CommonRoadGymEnv]
