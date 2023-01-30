from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Tuple
from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin

import numpy as np
from gym.spaces import Box

from crgeo.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from crgeo.common.class_extensions.string_resolver_mixing import StringResolverMixin

if TYPE_CHECKING:
    from crgeo.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class BaseControlSpaceOptions:
    pass


class BaseControlSpace(ABC, SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    Base class for facilitating learning-based agents' interaction with the traffic scene
    via the application of control signals on the simulated ego vehicle. The step and _substep
    methods allow for actions of any granularity to be supported, ranging from low-level control signals 
    to high-level maneuver planning.
    """

    def __init__(
        self,
        options: BaseControlSpaceOptions
    ) -> None:
        self._options = options
        self._steps_since_reset = 0

    @property
    @abstractmethod
    def gym_action_space(self) -> Box:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.gym_action_space.shape

    @property
    def steps_since_reset(self) -> int:
        return self._steps_since_reset

    def step(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
    ) -> Iterable[None]:
        """
        Represents a high-level action.
        """
        num_substeps = 0
        self._steps_since_reset += 1
        while 1:
            action_completed = self._substep(
                ego_vehicle_simulation=ego_vehicle_simulation,
                action=action,
                substep_index=num_substeps
            )
            num_substeps += 1
            yield
            if action_completed:
                break

    @abstractmethod
    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:
        """
        Abstract method executing the low-level control signal given by
        the current action. Returns True when the action is completed.
        """
        ...

    def reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        self._steps_since_reset = 0
        return self._reset(
            ego_vehicle_simulation=ego_vehicle_simulation
        )

    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        ...

