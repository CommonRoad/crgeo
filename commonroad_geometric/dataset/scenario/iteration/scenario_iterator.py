from __future__ import annotations

import logging
import os
from pathlib import Path
from random import Random
from typing import Callable, Iterable, Iterator, List, Optional

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.io_extensions.scenario_files import ScenarioFileFormat, find_scenario_paths
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.workers.async_preprocessing_worker import AsyncPreprocessingWorker
from commonroad_geometric.dataset.scenario.iteration.workers.base_preprocessing_worker import T_PreprocessingWorker
from commonroad_geometric.dataset.scenario.iteration.workers.sync_preprocessing_worker import SyncPreprocessingWorker
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor

logger = logging.getLogger(__name__)


class ScenarioIterationError(RuntimeError):
    pass


# Implemented as a Generator (yield from __iter__)
# See https://docs.python.org/3.10/library/typing.html#typing.Generator for suggestion to annotate it as an Iterable
class ScenarioIterator(Iterable[ScenarioBundle], AutoReprMixin):
    """
    Class for iterating over CommonRoad scenarios in a directory.

    Supports .xml scenario files and .pkl scenario bundles.
    ScenarioFileFormat.ALL will first search for the .pkl scenario bundles, then
    additionally search for '.xml' scenario files for which pickles do not exist yet.

    Preprocesses and filters with the BaseScenarioProcessor interface.

    Preprocessing: Action - Behavior
    Wrap with LogExceptionWrapper - Suppresses and logs exceptions
    Include ScenarioBundleWriter - Writes scenario bundles to pickle files
    Include ScenarioFormatConverter - Converts and saves scenarios to the desired output format during preprocessing

    Example:
    .. code-block:: python

        simple_scenario_iterator = ScenarioIterator(
            directory=Path('data', 'highd-sample'),
        )
        for scenario in scenario_iterator:
            print(scenario)

        complex_scenario_iterator = ScenarioIterator(
            directory=Path('data', 'highd-proto'),
            file_format = ScenarioFileFormat.XML,
            preprocessor = CutLeafLaneletsPreprocessor() >> CycleFilter() >> ScenarioBundleWriter(),
            is_looping = True,
            seed=42,
            workers=os.cpu_count(),
            cache_size=16,
        )
        for scenario in scenario_iterator:
            print(scenario)
    """

    def __init__(
        self,
        directory: Path,
        file_format: ScenarioFileFormat = ScenarioFileFormat.XML,
        filter_scenario_paths: Callable[[List[Path]], List[Path]] = lambda f: f,
        preprocessor: Optional[BaseScenarioPreprocessor] = None,
        is_looping: bool = False,
        seed: Optional[int] = None,
        workers: int = 1,
        cache_size: int = 8,
    ) -> None:
        """
        Creates a new ScenarioIterator using a synchronous, deterministic preprocessing worker when max_workers <= 1
        and an asynchronous, non-deterministic preprocessing worker when max_workers > 1.

        Args:
            directory (str): Base directory to load scenarios from, recursively includes subdirectories.
            file_format (ScenarioFileFormat):
                File extension to search for, supports: {'.xml', '.proto', '.pkl'}. Defaults to '.xml'.
            filter_scenario_paths (Callable[[List[Path]], List[Path]]):
                Callable which filters a list of paths, returning a sublist.
                See commonroad_geometric.common.io_extensions.scenario_files for implementations.
                Use in conjunction with functools.partial.
            preprocessor: (BaseScenarioProcessor):
                BaseScenarioPreprocessor based preprocessing pipeline.
                Defaults to IdentityPreprocessor (no preprocessing).
            is_looping (bool): If True, infinitely loops through found scenario paths. Defaults to False.
            seed (Optional[int]) Optional random seed, if given, randomly shuffles scenario paths.
            workers (int): How many preprocessing workers to use. Iteration non-deterministic for max_workers > 1.
            cache_size (int):
                How many scenarios should be preprocessed and cached before blocking the preprocessing worker.
                Ignored for workers <= 1.

        Raises:
            FileNotFoundError: If directory does not exist or is empty for file_format.
        """

        directory = Path(directory).resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory.absolute()} does not exist! "
                                    f"Make sure that the specified path leads to a scenario file or a directory.")

        if directory.is_file():
            self._scenario_paths = [directory]
        else:
            self._scenario_paths = find_scenario_paths(directory, file_format)

        self._scenario_paths = filter_scenario_paths(self._scenario_paths)

        if len(self._scenario_paths) == 0:
            raise FileNotFoundError(f"Directory {directory.absolute()} contains no scenario files with file extension: "
                                    f"{file_format}!")

        logger.info(f"Initialized with {len(self._scenario_paths)} scenario_paths")
        self._directory = directory

        if seed is not None:
            self._shuffle_scenario_paths(seed)

        preprocessor = IdentityPreprocessor() if preprocessor is None else preprocessor
        self._preprocessing_worker: T_PreprocessingWorker
        if workers <= 1:
            logger.info(f"Using synchronous preprocessing worker")
            self._preprocessing_worker = SyncPreprocessingWorker(
                preprocessor=preprocessor,
                is_looping=is_looping
            )
        else:
            logger.info(f"Using asynchronous preprocessing worker, with {workers} worker processes and cache size "
                        f"{cache_size}")
            self._preprocessing_worker = AsyncPreprocessingWorker(
                preprocessor=preprocessor,
                is_looping=is_looping,
                max_workers=workers,
                results_queue_size=cache_size,
            )

        logger.debug(f"Starting preprocessing worker")
        self._preprocessing_worker.start()

        # Attributes for iteration state
        self._result_iterator = None  # Iterator over preprocessing results
        self._current_scenario_bundles = []  # List of scenario bundles from the current preprocessing result
        self._current_bundle_index = 0  # Index within the current scenario bundles

    def _shuffle_scenario_paths(self, seed: int = None) -> None:
        """
        Shuffles internal scenario paths.

        Args:
            seed (int): seed for random number generator
        """
        rng = Random(seed)
        logger.info(f"Shuffling scenario_paths with seed {seed}")
        rng.shuffle(self._scenario_paths)

    @property
    def scenario_paths(self) -> List[Path]:
        return self._scenario_paths

    @property
    def preprocessor(self) -> BaseScenarioPreprocessor:
        return self._preprocessing_worker.preprocessor

    @property
    def is_looping(self) -> bool:
        return self._preprocessing_worker.is_looping

    @property
    def result_scenarios_per_scenario(self) -> int:
        return self._preprocessing_worker.preprocessor.results_factor

    @property
    def max_result_scenarios(self) -> int:
        return len(self._scenario_paths) * self.result_scenarios_per_scenario

    def __iter__(self) -> 'ScenarioIterator':
        """
        Returns:
            Self as an iterator over preprocessed ScenarioBundle's retrieved from preprocessing worker
        """
        self._preprocessing_worker.request_preprocessing(scenario_paths=self._scenario_paths)
        # Stops when get_next_preprocessing_result returns sentinel None
        self._result_iterator = iter(self._preprocessing_worker.get_next_preprocessing_result, None)
        self._current_scenario_bundles = []
        self._current_bundle_index = 0
        return self

    def __next__(self) -> ScenarioBundle:
        """
        Returns:
            The next ScenarioBundle from the preprocessing worker
        """
        # Check if there are more bundles in the current preprocessing result
        if self._current_bundle_index < len(self._current_scenario_bundles):
            scenario_bundle = self._current_scenario_bundles[self._current_bundle_index]
            self._current_bundle_index += 1
            return scenario_bundle

        # Try to get the next preprocessing result
        try:
            preprocessing_result = next(self._result_iterator)
        except StopIteration:
            self._preprocessing_worker.shutdown()
            raise StopIteration

        if preprocessing_result is None:
            # No more results; shutdown the worker and raise StopIteration
            self._preprocessing_worker.shutdown()
            raise StopIteration

        # Update the current scenario bundles and reset the index
        self._current_scenario_bundles = preprocessing_result
        self._current_bundle_index = 0

        # Recursively call __next__ to return the next bundle
        return self.__next__()