from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Union

import gymnasium.spaces
import numpy as np

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

T_Observation = Union[Dict[str, np.ndarray], np.ndarray]


class BaseObserver(ABC, SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    Base class for returning observations of the current traffic environment
    intended for RL agents. Implementations must implement the observe method,
    which must return a numpy array or a dictionary of such (for reasons of compatibility with StableBaselines3).
    """

    @abstractmethod
    def setup(self, dummy_data: CommonRoadData) -> gymnasium.Space:
        ...

    @abstractmethod
    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        ...

    def reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ):
        pass

    def debug_dict(
        self,
        observation: T_Observation
    ) -> Dict[str, str]:
        """
        Customizable extra debug overlays for rendering for inspecting the state space.

        Args:
            observation (T_Observation): Observation to inspect

        Returns:
            Dict[str, str]: Dictionary of info strings
        """
        return {}