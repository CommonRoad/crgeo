import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

class BaseRewardAggregator(ABC, AutoReprMixin, StringResolverMixin):
    """
    Base class for aggregating reward components.
    """

    _nan_warnings: Set[str] = set()
    
    def __init__(
        self,
        reward_computers: List[BaseRewardComputer]
    ) -> None:
        if len(reward_computers) == 0:
            warnings.warn("No reward computers provided. Default behavior: aggregate with zero rewards.")


        self._reward_computers = reward_computers
        self._reward_cache: Dict[BaseRewardComputer, float] = {}
        self._reward_component_info_step: Dict[str, float] = {}
        self._cumulative_reward: float = 0.0
        self._cumulative_step_reward: float = 0.0
        self._substep_reward: float = 0.0
        self._min_reward_step: float = float('inf')
        self._max_reward_step: float = -float('inf')
        self._substep_counter: int = 0
        self._highest: Optional[Tuple[str, float]] = None
        self._lowest: Optional[Tuple[str, float]] = None

    def reset_step(
        self
    ) -> None:
        self._cumulative_step_reward = 0.0
        self._substep_counter = 0
        self._min_reward_step = float('inf')
        self._max_reward_step = -float('inf')
        self._reward_component_info_step = {}
        
    @property
    def cumulative_reward_step(self) -> float:
        return self._cumulative_step_reward
        
    @property
    def avg_reward_step(self) -> float:
        return self._cumulative_step_reward / self._substep_counter if self._substep_counter > 0 else 0.0

    @property
    def min_reward_step(self) -> float:
        return self._min_reward_step
        
    @property
    def max_reward_step(self) -> float:
        return self._max_reward_step
        
    @property
    def reward_substep(self) -> float:
        return self._substep_reward

    @property
    def reward_component_info_step(self) -> Dict[str, float]:
        return self._reward_component_info_step

    def on_substep(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> None:
        reward_components: Dict[BaseRewardComputer, float] = {}
        reward_component_info: Dict[str, float] = {}

        self._substep_counter += 1

        lowest_reward = float('inf')
        highest_reward = -float('inf')
        lowest_reward_computer: Optional[str] = None
        highest_reward_computer: Optional[str] = None
        for computer in self._reward_computers:
            reward_component = computer.compute(
                action=action,
                simulation=simulation,
                data=data,
                observation=observation
            ) if simulation.current_time_step > simulation.initial_time_step + 1 else 0.0

            if not math.isfinite(reward_component):
                if type(computer).name not in BaseRewardAggregator._nan_warnings and not type(computer).allow_nan_values:
                    warnings.warn(f"Reward component {type(computer).name} has non-finite value {reward_component} at time-step {simulation.current_time_step}. Recurring warnings will be suppressed!") # type: ignore # TODO
                    BaseRewardAggregator._nan_warnings.add(type(computer).name)
                reward_component = self._reward_cache.get(computer, 0.0)

            if reward_component < lowest_reward:
                lowest_reward = reward_component
                lowest_reward_computer = computer.name
            if reward_component > highest_reward:
                highest_reward = reward_component
                highest_reward_computer = computer.name

            reward_components[computer] = reward_component
            reward_component_info[type(computer).name] = reward_component
            self._reward_cache[computer] = reward_component
        self._substep_reward = self._aggregate(reward_components)
        if self._substep_reward < self._min_reward_step:
            self._min_reward_step = self._substep_reward
        if self._substep_reward > self._max_reward_step:
            self._max_reward_step = self._substep_reward
        self._cumulative_reward += self._substep_reward
        self._cumulative_step_reward += self._substep_reward
        self._lowest = (lowest_reward_computer, lowest_reward)
        self._highest = (highest_reward_computer, highest_reward)
        for k, v in reward_component_info.items():
            if k not in self._reward_component_info_step:
                self._reward_component_info_step[k] = v
            else:
                self._reward_component_info_step[k] += v

    @abstractmethod
    def _aggregate(
        self,
        reward_components: Dict[BaseRewardComputer, float]
    ) -> float:
        ...

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def highest(self) -> Tuple[str, float]:
        return self._highest if self._highest is not None else ('', 0.0)

    @property
    def lowest(self) -> Tuple[str, float]:
        return self._lowest if self._lowest is not None else ('', 0.0)

    @property
    def highest_abs(self) -> float:
        return max(abs(self.highest[1]), abs(self.lowest[1]))
        
    def reset(self) -> None:
        self._reward_cache = {}
        self._cumulative_reward = 0.0
        self._highest = None
        self._lowest = None
        for computer in self._reward_computers:
            computer.reset()

    def component_aggregate_info(self) -> Dict[str, Dict[str, float]]:
        info: Dict[str, Dict[str, float]] = {}
        for computer in self._reward_computers:
            info[computer.name] = dict(
                max=computer.episode_max_reward,
                min=computer.episode_min_reward,
                sum=computer.episode_cumulative_reward,
                mean=computer.episode_mean_reward,
                abssum=computer.episode_cumulative_abs_reward,
                absmean=computer.episode_absmean_reward
                # TODO: Include statistics relative to highest_abs
            )
        return info
