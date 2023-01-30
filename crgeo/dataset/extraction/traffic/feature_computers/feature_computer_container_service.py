from __future__ import annotations

import logging
import warnings
from typing import Dict, Generic, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np
import torch

from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.common.torch_utils.helpers import is_finite
from crgeo.dataset.extraction.traffic.feature_computers.base_feature_computer import (BaseFeatureComputer, FunctionalFeatureComputer, T_FeatureParams)
from crgeo.dataset.extraction.traffic.feature_computers.types import FeatureDict, T_FeatureComputer
from crgeo.simulation.base_simulation import BaseSimulation, BaseSimulationOptions

logger = logging.getLogger(__name__)


class FeatureComputerContainerService(Generic[T_FeatureParams], AutoReprMixin):
    _nan_warnings: Set[str] = set()
    _compute_count: int = 0

    def __init__(
        self,
        feature_computers: List[T_FeatureComputer[T_FeatureParams]],
        suppress_feature_computer_exceptions: bool,
        simulation_options: BaseSimulationOptions,
        normalize: bool = False,
        stop_validation_after: Optional[int] = 200
    ) -> None:
        self._normalize = normalize
        self._suppress_feature_computer_exceptions = suppress_feature_computer_exceptions
        self._stop_validation_after = stop_validation_after

        self._feature_computers: List[BaseFeatureComputer[T_FeatureParams]] = []
        for feature_computer in feature_computers:
            if isinstance(feature_computer, BaseFeatureComputer):
                self._feature_computers.append(feature_computer)
            else:  # feature callable (T_FeatureCallable)
                self._feature_computers.append(FunctionalFeatureComputer[T_FeatureComputer](feature_computer)) # TODO: fix type

        # self.reset_all_feature_computers(None)
        self._setup_all_feature_computers(simulation_options)

        self._feature_column_indices: Dict[str, Tuple[int, int]]
        self._feature_dimension_total: int

    @property
    def num_computers(self) -> int:
        return len(self._feature_computers)

    @property
    def num_features(self) -> int:
        return len(self._feature_column_indices)

    @property
    def feature_column_indices(self) -> Dict[str, Tuple[int, int]]:
        return self._feature_column_indices

    @property
    def feature_dimensions(self) -> Dict[str, int]:
        dimensions: Dict[str, int] = {}
        for feature_computer in self._feature_computers:
            for feature_name, feature_dimension in feature_computer.feature_dimensions.items():
                dimensions[feature_name] = feature_dimension
        return dimensions

    def compute_all(
        self,
        jobs: Sequence[T_FeatureParams],
        simulation: BaseSimulation,
    ) -> torch.Tensor:
        if len(jobs) == 0:
            return torch.zeros((0, self._feature_dimension_total))

        x = torch.empty((len(jobs), self._feature_dimension_total), dtype=torch.float32)
        for i, params in enumerate(jobs):
            self.compute_row(params, simulation, out=x[i])

        if self._normalize:
            raise NotImplementedError("TODO")
            # for feature_name in self._features:
            #     if self._feature_types[feature_name] in (bool,):
            #         continue
            #     start_index, end_index = self.feature_column_indices[feature_name]
            #     x_column = x[:, start_index:end_index]
            #     if self._feature_normalizers[feature_name] is None:
            #         x[:, start_index:end_index] = x_column
            #     else:
            #         x_column_normalized = self._feature_normalizers[feature_name].normalize(arr=x_column, update=True)
            #         x[:, start_index:end_index] = x_column_normalized

        FeatureComputerContainerService._compute_count += 1

        return x

    def compute_row(
        self,
        params: T_FeatureParams,
        simulation: BaseSimulation,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        BaseFeatureComputer.reset_cache()

        if out is None:
            x_row = torch.empty((self._feature_dimension_total,), dtype=torch.float32)
        else:
            x_row = out

        for feature_computer in self._feature_computers:
            feature_dict: FeatureDict
            try:
                feature_dict = feature_computer.compute(params, simulation)
            except Exception as e:
                logger.exception("Feature computer %s threw an exception", feature_computer.name)
                if not self._suppress_feature_computer_exceptions:
                    raise e

                feature_dict = {}
                for feature_name, feature_dim in feature_computer.feature_dimensions.items():
                    if feature_dim > 1:
                        feature_dict[feature_name] = torch.full((feature_dim,), fill_value=np.nan)
                    else:
                        feature_dict[feature_name] = np.nan

            for feature_name, feature_value in feature_dict.items():
                if self._stop_validation_after is None or FeatureComputerContainerService._compute_count < self._stop_validation_after:
                    if all((
                        not feature_computer.__class__.allow_nan_values,
                        feature_name not in FeatureComputerContainerService._nan_warnings,
                        not is_finite(feature_value),
                    )):
                        warnings.warn(f"Feature {feature_name} ({feature_computer.name}) has non-finite value {feature_value} at time-step {params.time_step}. Recurring warnings will be suppressed!")
                        FeatureComputerContainerService._nan_warnings.add(feature_name)

                start_index, end_index = self.feature_column_indices[feature_name]

                x_row[start_index:end_index] = feature_value

        return x_row

    def reset_all_feature_computers(self, simulation: Optional[BaseSimulation]) -> None:
        for feature_computer in self._feature_computers:
            feature_computer.reset(simulation)

    def _setup_all_feature_computers(self, simulation_options: BaseSimulationOptions) -> None:
        """Caches dimension of each feature computer (needed for pre-allocation)"""

        # Resetting storage
        self._feature_column_indices = {}
        # self._feature_normalizers = {}
        self._feature_dimension_total = 0

        for feature_computer in self._feature_computers:
            feature_computer.setup(simulation_options)
            for feature_name, feature_dim in feature_computer.feature_dimensions.items():
                self._feature_column_indices[feature_name] = (
                    self._feature_dimension_total,
                    self._feature_dimension_total + feature_dim,
                )
                self._feature_dimension_total += feature_dim
