import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult

logger = logging.getLogger(__name__)


class _BasePreprocessingWorker(ABC, AutoReprMixin):
    def __init__(
        self,
        preprocessor: BaseScenarioPreprocessor,
        is_looping: bool
    ) -> None:
        self._preprocessor = preprocessor
        self._is_looping = is_looping

    @property
    def preprocessor(self) -> BaseScenarioPreprocessor:
        return self._preprocessor

    @property
    def is_looping(self) -> bool:
        return self._is_looping

    @abstractmethod
    def start(self) -> None:
        """
        Starts the preprocessing worker.
        Needs to be called before calling request_preprocessing() and get_next_preprocessing_results().
        """
        ...

    @abstractmethod
    def request_preprocessing(self, scenario_paths: Sequence[Path]) -> None:
        """
        Requests preprocessing from the worker for the specified scenario_paths.

        Args:
            scenario_paths (Sequence[Path]): scenario paths which should be preprocessed

        Raises:
            RuntimeError: If start() has not been called yet or if shutdown() was called
        """
        ...

    @abstractmethod
    def get_next_preprocessing_result(self) -> Optional[T_ScenarioPreprocessorResult]:
        """
        Gets next preprocessing result from worker.

        Returns:
            Either:
                1) next preprocessing result as soon as it is available
                2) None if no more preprocessing results are available
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shuts down the preprocessing worker.
        No other methods can be called after this one.
        """
        ...


T_PreprocessingWorker = _BasePreprocessingWorker
