from itertools import cycle
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.iteration.workers.base_preprocessing_worker import _BasePreprocessingWorker
from commonroad_geometric.dataset.scenario.iteration.workers.utils import preprocess_scenario


class SyncPreprocessingWorker(_BasePreprocessingWorker):
    """
    Synchronous, non-threaded, non-multiprocessing version of AsyncPreprocessingWorker.
    Much slower (~5x), should be used when determinism is necessary.
    E.g. for testing or easier debugging of dependent components.

    """

    def __init__(
        self,
        preprocessor: BaseScenarioPreprocessor,
        is_looping: bool
    ) -> None:
        self._requested_scenario_paths: Iterator[Path]
        self._is_alive: bool = False
        super().__init__(preprocessor, is_looping)

    def start(self) -> None:
        self._is_alive = True

    def request_preprocessing(self, scenario_paths: Sequence[Path]) -> None:
        if not self._is_alive:
            raise RuntimeError(f"Call to request_preprocessing() before calling start() or after calling shutdown()")

        if self._is_looping:
            self._requested_scenario_paths = cycle(scenario_paths)
        else:
            self._requested_scenario_paths = iter(scenario_paths)

    def get_next_preprocessing_result(self) -> Optional[T_ScenarioPreprocessorResult]:
        if not self._is_alive:
            raise RuntimeError(f"Call to get_next_preprocessing_result() before calling start() or after calling shutdown()")

        try:
            scenario_path = next(self._requested_scenario_paths)
            return preprocess_scenario(scenario_path, self._preprocessor)
        except StopIteration:
            return None

    def shutdown(self) -> None:
        self._requested_scenario_paths = cycle([])
        self._is_alive = False
