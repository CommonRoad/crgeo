from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Type, cast
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import load_dill, save_dill
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.learning.reinforcement.commonroad_gym_env import RLEnvironmentOptions, RLEnvironmentParams
from commonroad_geometric.learning.reinforcement.constants import COMMONROAD_GYM_ENV_ID
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.base_reward_aggregator import BaseRewardAggregator
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation_factory import EgoVehicleSimulationFactory
from commonroad_geometric.simulation.ego_simulation.respawning import BaseRespawner, BaseRespawnerOptions

EXPORT_FILENAME = 'experimentconfig'
EXPORT_FILETYPE = 'pkl'


@dataclass
class RLExperimentConfig:
    """
    simulation_cls: train agent in specified simulation
    simulation_options: options within simulation class
    control_space_cls: high level or low level control space of agent(e.g. highlevel(lane change) lowlevel(longitudinal control))
    control_space_options: options within control space class
    respawner_cls: BaseEgoVehicleRespawner class that specify how to respawn ego vehicle when a run is terminated
    respawner_options: options within respawner class
    traffic_extraction_options: options of traffic extractors, which obtains features for CommonRoadData
    rewarder: compute and aggregate reward
    termination_criteria: list of criterien that terminate the current run of agent and start respawning
    env_options: options for CommonRoadGymEnv
    """
    control_space_cls: Type[BaseControlSpace]
    control_space_options: BaseControlSpaceOptions
    ego_vehicle_simulation_options: EgoVehicleSimulationOptions
    env_options: RLEnvironmentOptions
    respawner_cls: Type[BaseRespawner]
    respawner_options: BaseRespawnerOptions
    rewarder: BaseRewardAggregator
    simulation_cls: Type[BaseSimulation]
    simulation_options: BaseSimulationOptions
    termination_criteria: List[BaseTerminationCriterion]
    traffic_extraction_options: TrafficExtractorOptions


class RLExperiment:
    def __init__(self, config: RLExperimentConfig) -> None:
        self.config = config

    def create_name(self) -> str:
        name = f"{COMMONROAD_GYM_ENV_ID}-{self.config.control_space_cls.__name__}-{get_timestamp_filename(include_time=False)}"
        return name

    def _setup_env_params(self, scenario_dir: Path,
                          override_options: Optional[RLEnvironmentOptions] = None) -> RLEnvironmentParams:
        ego_vehicle_simulation_factory = EgoVehicleSimulationFactory(
            simulation_cls=self.config.simulation_cls,
            simulation_options=self.config.simulation_options,
            control_space_cls=self.config.control_space_cls,
            control_space_options=self.config.control_space_options,
            respawner_cls=self.config.respawner_cls,
            respawner_options=self.config.respawner_options,
            extractor_factory=TrafficExtractorFactory(self.config.traffic_extraction_options),
            ego_vehicle_simulation_options=self.config.ego_vehicle_simulation_options
        )
        env_params = RLEnvironmentParams(
            ego_vehicle_simulation_factory=ego_vehicle_simulation_factory,
            scenario_dir=scenario_dir,
            rewarder=self.config.rewarder,
            termination_criteria=self.config.termination_criteria,
            options=override_options if override_options is not None else self.config.env_options
        )
        return env_params

    def make_env(
        self,
        scenario_dir: Path,
        n_envs: int = 1,
        seed: int = 0,
        **env_options_kwargs
    ) -> VecEnv:
        try:
            env_options = self.config.env_options
        except AttributeError:
            env_options = RLEnvironmentOptions()
        env_options_dict = {k: v for k, v in asdict(env_options).items()}
        env_options_dict.update(env_options_kwargs)
        env_params = self._setup_env_params(scenario_dir) #, override_options=RLEnvironmentOptions(**env_options_dict))
        vec_env_cls: Type[VecEnv] = DummyVecEnv if n_envs == 1 else SubprocVecEnv  # type: ignore # TODO
        env = make_vec_env(
            env_id=COMMONROAD_GYM_ENV_ID,
            n_envs=n_envs,
            seed=seed,
            env_kwargs=dict(params=env_params),
            vec_env_cls=vec_env_cls
        )
        return env

    @staticmethod
    def _get_file_path(directory: Path) -> Path:
        directory = Path(directory)
        return directory.joinpath(EXPORT_FILENAME + '.' + EXPORT_FILETYPE)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        experiment_path = self._get_file_path(directory)
        save_dill(self.config, experiment_path)
        return experiment_path

    @classmethod
    def load(cls, file_path: Path) -> RLExperiment:
        file_path = Path(file_path)
        file_path = cls._get_file_path(file_path) if not file_path.name.endswith(EXPORT_FILETYPE) else file_path
        config = cast(RLExperimentConfig, load_dill(file_path))
        experiment = RLExperiment(config)
        return experiment
