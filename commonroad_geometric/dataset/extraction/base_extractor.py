from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TYPE_CHECKING, TypeVar, Union

import torch
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal

if TYPE_CHECKING:
    from commonroad_geometric.simulation.base_simulation import BaseSimulation

T_BaseExtractorOptions = TypeVar("T_BaseExtractorOptions", bound="BaseExtractorOptions")
T_BaseExtractor = TypeVar("T_BaseExtractor", bound="BaseExtractor")
T_BaseExtractionParams = TypeVar("T_BaseExtractionParams", bound="BaseExtractionParams")
# There is nothing else right now
T_ExtractionReturnType = TypeVar('T_ExtractionReturnType', bound=Union[CommonRoadData, CommonRoadDataTemporal])


@dataclass
class BaseExtractorOptions:
    r"""
    Options for the extractor. Stays the same over several scenarios.
    """
    pass


@dataclass
class BaseExtractionParams:
    r"""
    Parameters for the extraction. Might change with every scenario.
    """
    disable_postprocessing: bool = False
    device: Optional[str | torch.device] = None


class BaseExtractor(
    ABC,
    Generic[T_BaseExtractorOptions, T_BaseExtractionParams, T_ExtractionReturnType],
    AutoReprMixin
):
    r"""
    Base class for extracting graphs from CommonRoad scenarios for downstream graph neural network learning.
    """

    def __init__(
        self,
        simulation: BaseSimulation,
        options: T_BaseExtractorOptions
    ) -> None:
        r"""
        Initializes the BaseExtractor with a simulation instance.

        Args:
            simulation (BaseSimulation): Simulation of CommonRoad scenario.
            options (T_BaseExtractorOptions): Options for extraction.
        """
        self._simulation = simulation
        self._options = options

    @property
    def simulation(self) -> BaseSimulation:
        return self._simulation

    @property
    def scenario(self) -> Scenario:
        return self._simulation.current_scenario

    @scenario.setter
    def scenario(self, value: Scenario) -> None:
        raise ValueError("To set the scenario of the extractor directly is prohibited! Use the simulation.")

    @property
    def options(self):
        return self._options

    @abstractmethod
    def extract(
        self,
        time_step: int,
        params: T_BaseExtractionParams,
    ) -> T_ExtractionReturnType:
        r"""
        Extracts customized graph representation of the current CommonRoad traffic environment and returns it as a
        CommonRoadData data instance.

        Args:
            time_step (int): Time step for which data instance should be extracted.
            params (T_BaseExtractionParams): Additional parameters for extracting the data instance.

        Returns:
            Data instance.
        """
        ...

    @abstractmethod
    def __len__(self) -> T_CountParam:
        r"""
        Returns:
            The number of samples that can be extracted from the provided simulation.
        """

    @abstractmethod
    def __iter__(self) -> BaseExtractor:
        r"""
        Returns:
            This extractor as an iterator.
        """
        ...

    @abstractmethod
    def __next__(self) -> T_ExtractionReturnType:
        r"""
        Returns:
            The next extracted data instance retrieved from this extractor as an iterator.
        """
        ...
