from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TYPE_CHECKING, TypeVar

from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.rendering import Renderable

if TYPE_CHECKING:
    from commonroad_geometric.simulation.base_simulation import BaseSimulation

TypeVar_BaseExtractorOptions = TypeVar('TypeVar_BaseExtractorOptions', bound='BaseExtractorOptions')
TypeVar_BaseExtractor = TypeVar('TypeVar_BaseExtractor', bound='BaseExtractor')


@dataclass
class BaseExtractorOptions:
    pass


@dataclass
class BaseExtractionParams:
    """
    Args:
        index (int): E.g. time-step or edge index
    """
    index: int
    disable_postprocessing: bool = False


T_BaseExtractionParams = TypeVar("T_BaseExtractionParams", bound=BaseExtractionParams)
T_ExtractionReturnType = TypeVar('T_ExtractionReturnType')


class BaseExtractor(ABC, AutoReprMixin, Generic[T_BaseExtractionParams, T_ExtractionReturnType]):
    """Base class for facilitating graph extractions from CommonRoad scenarios to facilitate
    downstream graph neural network learning.
    """

    def __init__(
        self,
        simulation: BaseSimulation,
        options: BaseExtractorOptions
    ) -> None:
        """

        Args:
            simulation (BaseSimulation): Simulation of CommonRoad scenario.
        """
        self._simulation = simulation

    @property
    def simulation(self) -> BaseSimulation:
        return self._simulation

    @property
    def scenario(self) -> Scenario:
        return self._simulation.current_scenario

    @scenario.setter
    def scenario(self, value: Scenario) -> None:
        raise ValueError("To set the scenario of the extractor directly is prohibited! Use the simulation.")

    @abstractmethod
    def extract(self, params: T_BaseExtractionParams) -> T_ExtractionReturnType:
        """Extracts customized graph representation of the current CommonRoad traffic
        environment and returns it as a CommonRoadData instance.
        """

    @abstractmethod
    def __len__(self) -> T_CountParam:
        """Returns the number of samples that can be extracted from the provided scenario.
        """

    @abstractmethod
    def __iter__(self) -> BaseExtractor:
        ...

    @abstractmethod
    def __next__(self) -> T_ExtractionReturnType:
        ...
