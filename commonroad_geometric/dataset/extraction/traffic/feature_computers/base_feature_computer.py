from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

import numpy as np
import torch
from torch import Tensor

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.class_property_decorator import classproperty
from commonroad_geometric.common.class_extensions.safe_pickling_mixin import SafePicklingMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.common.utils.functions import get_function_return_variable_name
from commonroad_geometric.dataset.extraction.traffic.exceptions import TrafficExtractionException
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import BaseFeatureParams, FeatureDict, FeatureValue, T_FeatureCallable, T_FeatureParams
from commonroad_geometric.dataset.extraction.traffic.feature_computers.utils.init_mock_setup import _init_mock_setup
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation


class InvalidFeatureValueException(TrafficExtractionException):
    pass


class InvalidFeatureNameException(TrafficExtractionException):
    pass


class DuplicateFeatureNameException(TrafficExtractionException):
    def __init__(self, feature_name: str, old_computer_name: str, new_computer_name: str) -> None:
        super().__init__(f"Encountered duplicate feature '{feature_name}' from feature computer {new_computer_name} (already returned by {old_computer_name})")


class FeatureComputerNotSetupException(TrafficExtractionException):
    pass


class BaseFeatureComputer(ABC, Generic[T_FeatureParams], SafePicklingMixin, AutoReprMixin, StringResolverMixin):
    """
    Base class for computing features to be included as attributes in
    the Data instances originating from 'TrafficExtractor'.

    Feature computer is easily customizable by writing child class of BaseFeatureComputer and overwrite abstractmethod __call__  
    Note to add new features to commonroad_geometric/dataset/extraction/traffic/feature_computers/implementations/types.py
    """

    ComputedFeaturesCache: Dict[str, FeatureValue] = {}
    _MockSimulation: Optional[ScenarioSimulation] = None
    _MockParams: Optional[ List[BaseFeatureParams]] = None
    _FeatureComputerMap: Dict[str, BaseFeatureComputer[Any]] = {}
    _FeatureDimensions: Dict[str, Dict[str, int]] = {}
    _FeatureTypes: Dict[str, Dict[str, type]] = {}
    _FeatureDimensionsTotal: Dict[str, int] = {}
    _FeatureNames: Dict[str, List[str]] = {}

    def __init__(self) -> None:
        self._name = self._get_name()
        self._activated: bool = False

    def _get_name(self) -> str:
        return type(self).__name__

    @classproperty
    def allow_nan_values(cls) -> bool:
        return False

    @classproperty
    def skip_normalize_features(cls) -> Optional[Set[str]]:
        return None

    @property
    def feature_dimensions(self) -> Dict[str, int]:
        if not self.is_setup:
            raise FeatureComputerNotSetupException()
        return self._FeatureDimensions[self.name]

    @property
    def feature_types(self) -> Dict[str, type]:
        if not self.is_setup:
            raise FeatureComputerNotSetupException()
        return self._FeatureTypes[self.name]

    @property
    def feature_dimension_total(self) -> int:
        if not self.is_setup:
            raise FeatureComputerNotSetupException()
        return self._FeatureDimensionsTotal[self.name]

    @property
    def features(self) -> List[str]:
        if not self.is_setup:
            raise FeatureComputerNotSetupException()
        return  self._FeatureNames[self.name]

    @property
    def is_setup(self) -> bool:
        return self.name in self._FeatureDimensions

    def setup(self, simulation_options: BaseSimulationOptions):
        """
        Computes features on mock data in order to derive feature types and dimensions.

        Raises:
            DuplicateFeatureNameException: Same feature name used more than once.
        """
        if self.is_setup:
            return

        if self._MockSimulation is None or self._MockParams is None:
            self._MockSimulation, self._MockParams = _init_mock_setup(simulation_options)

        self._FeatureTypes[self.name] = {}
        self._FeatureDimensions[self.name] = {}
        self._FeatureNames[self.name] = []
        self._FeatureDimensionsTotal[self.name] = 0

        for i, mock in enumerate(self._MockParams):
            feature_dict = self.compute(mock, self._MockSimulation)
            if i == 0:
                for feature_name, feature_value in feature_dict.items():
                    if feature_name in self._FeatureComputerMap:
                        raise DuplicateFeatureNameException(
                            feature_name,
                            old_computer_name=self._FeatureComputerMap[feature_name].name,
                            new_computer_name=self.name
                        )
                    self._FeatureComputerMap[feature_name] = self
                    self._validate_feature(feature_name, feature_value)
                    self._FeatureTypes[self.name][feature_name] = type(feature_value)
                    feature_value_dim = self._get_feature_value_dim(feature_value)
                    self._FeatureDimensions[self.name][feature_name] = feature_value_dim
                    self._FeatureNames[self.name].append(feature_name)
                    self._FeatureDimensionsTotal[self.name] += feature_value_dim

    @classmethod
    def reset_cache(cls) -> None:
        """
        To be called at each time-step.
        """
        cls.ComputedFeaturesCache = {}

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        params: T_FeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        """
        The compute method is being called for each item at each time-step, and returns the computed features.

        Args:
            params (T_FeatureParams):
                Struct containing parameters which can be used to compute the features.
            simulation:
                BaseSimulation that keeps track of the traffic scene, allowing e.g. lanelet lookup for obstacles.

        Returns:
            FeatureDict:
                Dictionary mapping from feature name to feature values of valid types,
                i.e. either floats, integers, booleans or 1-d PyTorch tensors.
        """
        if not self._activated:
            self.reset(simulation)
        self._activated = True
        feature_dict_raw = self(params, simulation)
        feature_dict: FeatureDict = {}

        for key, raw_value in feature_dict_raw.items():
            if isinstance(raw_value, np.ndarray):
                value = torch.from_numpy(raw_value).to(torch.float32)
            else:
                value = raw_value
            if isinstance(value, Tensor) and value.ndim > 1:
                value = value.flatten()
            self.ComputedFeaturesCache[key] = value
            feature_dict[key] = value

        return feature_dict

    @abstractmethod
    def __call__(
        self,
        params: T_FeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        ...

    def reset(self, simulation: Optional[BaseSimulation]) -> None:
        """
        The reset method resets the internal state of a feature computer and prepares.
        It is called before processing a new scenario.
        """
        simulation = simulation or BaseFeatureComputer._MockSimulation
        return self._reset(simulation)

    def _reset(self, simulation: BaseSimulation) -> None:
        pass

    @staticmethod
    def _get_feature_value_dim(feature_value: FeatureValue) -> int:
        if isinstance(feature_value, (float, int, bool)):
            return 1
        elif isinstance(feature_value, (torch.Tensor, np.ndarray)):
            if feature_value.ndim == 0:
                return 1
            else:
                return feature_value.shape[0]
        else:
            raise NotImplementedError(type(feature_value))

    @staticmethod
    def _validate_feature(feature_name: str, feature_value: Any) -> None:
        if not isinstance(feature_name, str):
            raise InvalidFeatureNameException(f"Encountered feature_name {feature_name} ({type(feature_name)}). Only feature names of type str are allowed.")
        if not isinstance(feature_value, (float, int, bool, torch.Tensor, np.ndarray)):
            raise InvalidFeatureValueException(f"Feature extractor {feature_name} returned invalid type {type(feature_value)}.")
        if isinstance(feature_value, (torch.Tensor, np.ndarray)) and feature_value.ndim > 1:
            raise InvalidFeatureValueException(f"Feature computer {feature_name} returns a tensor with {feature_value.ndim} dimensions. "
                                               f"Only one or zero-dimensional tensors are allowed.")


class FunctionalFeatureComputer(BaseFeatureComputer[T_FeatureParams], Generic[T_FeatureParams]):
    """Wrapper class for a stateless feature callable"""

    def __init__(self, feature_computer: T_FeatureCallable[T_FeatureParams]) -> None:
        self._feature_computer: T_FeatureCallable[T_FeatureParams] = feature_computer
        signature = inspect.signature(feature_computer)
        self._accepts_simulation = len(signature.parameters) == 2
        super(FunctionalFeatureComputer, self).__init__()

    def _get_name(self) -> str:
        if self._feature_computer.__name__ == "<lambda>":
            return 'lambda-' + get_function_return_variable_name(self._feature_computer)
        return self._feature_computer.__name__

    def __call__(
        self,
        params: T_FeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        if self._accepts_simulation:
            return self._feature_computer(params, simulation) # TODO: fix type issues
        else:
            return self._feature_computer(params)
