from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.common.utils.string import rchop
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.observer.base_observer import T_Observation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class BaseRewardComputer(ABC, SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    Base class for computing reward components.
    """

    def __init__(self) -> None:
        self._cumulative: float
        self._cumulative_abs: float
        self._max: float
        self._min: float
        self._call_count: int
        self.reset()

    @property
    def episode_max_reward(self) -> float:
        return self._max

    @property
    def episode_min_reward(self) -> float:
        return self._min

    @property
    def episode_cumulative_reward(self) -> float:
        return self._cumulative

    @property
    def episode_cumulative_abs_reward(self) -> float:
        return self._cumulative_abs

    @property
    def episode_mean_reward(self) -> float:
        return self.episode_cumulative_reward / self._call_count if self._call_count > 0 else np.nan

    @property
    def episode_absmean_reward(self) -> float:
        return self.episode_cumulative_abs_reward / self._call_count if self._call_count > 0 else np.nan

    @classproperty
    def name(cls) -> str:
        return rchop(cls.__name__, 'RewardComputer')  # type: ignore

    @classproperty
    def allow_nan_values(cls) -> bool:
        return False

    def compute(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        """
        The __call__ method returns the computed reward component.

        Args:
            data (CommonRoadData):
                Traffic graph data instanc for the current time-step.

        Returns:
            float:
                Reward signal
        """
        reward = self(
            action=action,
            simulation=simulation,
            data=data,
            observation=observation
        )
        self._cumulative += reward
        self._cumulative_abs += abs(reward)
        if reward > self._min:
            self._min = reward
        if reward < self._max:
            self._max = reward
        self._call_count += 1
        return reward

    @abstractmethod
    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        """
        The __call__ method returns the computed reward component.

        Args:
            data (CommonRoadData):
                Traffic graph data instanc for the current time-step.

        Returns:
            float:
                Reward signal
        """

    def reset(self) -> None:
        self._cumulative = 0.0
        self._cumulative_abs = 0.0
        self._max = float('inf')
        self._min = -float('inf')
        self._call_count = 0
        self._reset()

    def _reset(self) -> None:
        pass
