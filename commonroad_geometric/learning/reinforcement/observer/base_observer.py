from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Union
import gym.spaces

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation

T_Observation = Union[Dict[str, np.ndarray], np.ndarray]

class BaseObserver(ABC, SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    Base class for returning observations of the current traffic environment
    intended for RL agents. Implementations must implement the observe method,
    which must return a numpy array or a dictionary of such (for reasons of compatibility with StableBaselines3). 
    """

    @abstractmethod
    def setup(self, dummy_data: CommonRoadData) -> gym.spaces.Space:
        ...

    @abstractmethod
    def observe(
        self,
        data: CommonRoadData,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Observation:
        ...
