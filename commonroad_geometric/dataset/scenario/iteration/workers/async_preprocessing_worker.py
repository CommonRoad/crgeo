import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.iteration.workers.base_preprocessing_worker import _BasePreprocessingWorker
from commonroad_geometric.dataset.scenario.iteration.workers.preprocessing_thread import _PreprocessingThread

logger = logging.getLogger(__name__)


class AsyncPreprocessingWorker(_BasePreprocessingWorker):
    """
    This class handles the partially asynchronous, complex loading and preprocessing of scenarios.
    All methods of this class are only called by the main thread.

    This worker spawns a background thread which uses a concurrent.futures.ProcessPoolExecutor
    to read scenario files and run preprocessing asynchronously.
    """

    def __init__(
        self,
        preprocessor: BaseScenarioPreprocessor,
        is_looping: bool,
        max_workers: int,
        results_queue_size: int = 0,  # 0 means infinite
    ) -> None:
        """
        Warning: The state of the preprocessor is only available to the process it is run in by the ProcessPoolExecutor!

        Args:
            preprocessor (BaseScenarioPreprocessor): preprocessor used by the ProcessPoolExecutor
            is_looping (bool): If True, infinitely loops through most recently requested scenario paths. Defaults to False.
            max_workers (int): How many processes the ProcessPoolExecutor will create
            results_queue_size (int): Size of the results queue, blocks workers when it is full
        """
        # Composition over inheritance to:
        #   a) avoid multiple inheritance (i.e. avoid dynamic dispatch issues/overriding methods)
        #   b) easily implement a synchronous version with the same interface
        #   c) separate code "of" the main thread from the worker thread
        super(AsyncPreprocessingWorker, self).__init__(
            preprocessor=preprocessor,
            is_looping=is_looping
        )
        self._thread = _PreprocessingThread(
            preprocessor=self.preprocessor,
            is_looping=self.is_looping,
            workers=max_workers,
            result_queue_size=results_queue_size
        )

    def start(self) -> None:
        self._thread.start()

    def request_preprocessing(self, scenario_paths: Sequence[Path]) -> None:
        if not self._thread.is_alive():
            raise RuntimeError(f"Call to request_preprocessing() before calling start() or after calling shutdown()")

        logger.info(f"Try requesting preprocessing for {len(scenario_paths)} scenario_paths")
        self._thread.request_preprocessing(scenario_paths)
        logger.info(f"Requested preprocessing for {len(scenario_paths)} scenario_paths")

    def get_next_preprocessing_result(self) -> Optional[T_ScenarioPreprocessorResult]:
        if not self._thread.is_alive():
            raise RuntimeError(f"Call to get_next_preprocessing_result() before calling start() or after calling shutdown()")

        logger.debug(f"Try getting preprocessing result")
        result = self._thread.get_preprocessing_result()
        if result is None:
            logger.debug(f"Got sentinel value None from scenario_path_queue, attempting to shutdown, returning None")
            self.shutdown()
            return None
        logger.debug(f"Got preprocessing result={result}")
        return result

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        thread_thread_id = self._thread.native_id
        self._thread.shutdown()
        logger.info(f"Waiting for background {self._thread.name}(thread_id={thread_thread_id}) to join thread_id={threading.get_native_id()}")
        self._thread.join()
        logger.info(f"Joined background {self._thread.name}(thread_id={thread_thread_id}) in thread_id={threading.get_native_id()}")
