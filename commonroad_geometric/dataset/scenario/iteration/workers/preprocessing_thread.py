import logging
import threading
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Optional, Sequence, Type

from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.iteration.workers.utils import preprocess_scenario

logger = logging.getLogger(__name__)


class _ShutdownQueue:
    """
    Class used as a sentinel value.
    When this class is retrieved from a queue, the queue should be closed.
    Replace with https://peps.python.org/pep-0661/ if approved.

    More information:
        https://stackoverflow.com/questions/69239403/type-hinting-parameters-with-a-sentinel-value-as-the-default
    """
    pass


class _PreprocessingThread(Thread):
    """
    Uses the producer-consumer pattern with queue.Queue (synchronized for the threading module).
    Creates two queues, which are used as follows to communicate with the main thread.

    Scenario path queue ("input queue"):
        Producer 1: request_preprocessing puts all scenario_paths in the scenario_path_queue
        Consumer 1: Consume a scenario_path, submit the preprocessing-future to the ProcessPoolExecutor

    Preprocessing bundle queue ("output queue")
        Producer 2: The "done_callback" attached to the preprocessing-future, puts the *result* of the future in the preprocessing_results_queue
        Consumer 2: get_preprocessing_results gets the *result* from the preprocessing_results_queue as soon as one is available

    Mandatory reading before doing anything related to Python multiprocessing:
        https://docs.python.org/3/library/multiprocessing.html#programming-guidelines
        https://docs.python.org/3.10/library/multiprocessing.html#contexts-and-start-methods

    The first thing I would improve is using a cancellable, clearable Queue implementation, ideally with thread-safe task counting.
    Right now this clearing is hacked on top of the standard Queue implementation by accessing its internal locks.
    Thread-safe task counting is done with a Semaphore, as the unfinished_tasks attribute of the Queue is not part of the public API (see https://stackoverflow.com/a/36658750).

    See the following as a warning before you attempt to modify this.
    Here are some of my other attempts at implementing this class:

    1)
    multiprocessing.Queue

        Can't implement this with concurrency.futures.ProcessPoolExecutor and multiprocessing.Queue,
        as Queue's can't be shared between processes, leading to:

        > RuntimeError: Queue objects should only be shared between processes through inheritance

        Careful: Here inheritance refers to the child process inheriting the memory of the parent process!

    2)
    multiprocessing.Manager and multiprocessing.Manager.Queue

        Instead, we *should* be able to use the multiprocessing.managers.SyncManager.Queue
        which creates a proxy for the Queue, which can be shared between processes.
        However, this also fails as soon as we try to use the proxy Queue as
        an instance variable, i.e. using at as: self.queue = manager.Queue().
        The specific error is usually:

        >   File ".../anaconda3/envs/commonroad-3.10/lib/python3.10/multiprocessing/managers.py", line 810, in _callmethod
        >     conn = self._tls.connection
        > AttributeError: 'ForkAwareLocal' object has no attribute 'connection'

        This would work if we would keep the proxy Queue either as:
            a) a global variable
            b) outside a class (preventing us from using it in a method of a class)

        I.e. this is completely useless outside a short Python script where we create the Queue in the main function

    3)
    Switch concurrency.futures.ProcessPoolExecutor for multiprocessing.Pool, unrelated to 1) and 2)

        I am confident we could implement this with either ProcessPoolExecutor and Pool.
        ProcessPoolExecutor returns a Future. Pool returns an AsyncResult.
        The advantage of ProcessPoolExecutor is that we can cancel futures.
        More information: https://docs.python.org/3.10/library/multiprocessing.html

    4)
    multiprocessing.Semaphore and multiprocessing.Manager.Semaphore

        Idea: Pass all paths as a list and use a semaphore to limit how many paths are processed.
        Issues: Same as 1) and 2), as well as losing the convenience of a Queue,
        e.g. being able to add more tasks at a later point, easily accessing the results

    More information on 1) and 2):
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager.Queue
        https://stackoverflow.com/questions/43439194/python-multiprocessing-queue-vs-multiprocessing-manager-queue
        https://towardsdatascience.com/pool-limited-queue-processing-in-python-2d02555b57dc
        https://stackoverflow.com/questions/50869656/multiple-queues-from-one-multiprocessing-manager
        https://stackoverflow.com/questions/3217002/how-do-you-pass-a-queue-reference-to-a-function-managed-by-pool-map-async
        https://stackoverflow.com/questions/26225108/do-i-need-to-explicitly-pass-multiprocessing-queue-instance-variables-to-a-child

    """

    def __init__(
        self,
        preprocessor: BaseScenarioPreprocessor,
        is_looping: bool,
        workers: int,
        result_queue_size: int,
    ) -> None:
        """
        Runs in:
            Only the main thread.

        Args:
            preprocessor (BaseScenarioPreprocessor): preprocessor applied to requested scenario_paths
            is_looping (bool): Whether to infinitely loop through the most recently requested scenario paths. Defaults to False.
            workers (int): How many processes the ProcessPoolExecutor will create
            result_queue_size (int): Size of the preprocessing_results_queue, blocks workers when it is full
        """
        self._preprocessor = preprocessor
        self._is_looping = is_looping
        self._max_workers = workers

        self._scenario_path_queue: Queue[Path | Type[_ShutdownQueue]] = Queue()
        self._preprocessing_result_queue: Queue[T_ScenarioPreprocessorResult | Type[_ShutdownQueue]] = Queue(maxsize=result_queue_size)
        self._kill_switch = Event()

        # Task-count house-keeping is necessary to enable looping
        self._remaining_task_counter = threading.Semaphore(value=0)
        # Keeping track of the last request scenario paths is necessary to enable looping
        self._last_requested_scenario_paths: Sequence[Path]

        self._executor: ProcessPoolExecutor

        super(_PreprocessingThread, self).__init__(
            name=type(self).__name__,
            daemon=True  # This needs to be a daemon thread, so that the program exits when the main thread ceases to exist
        )

        # thread_id==None indicates that Thread has not been started yet
        logger.debug(f"Initialized {self.name}(thread_id={self.native_id}) with preprocessor={self._preprocessor.name}")

    def run(self) -> None:
        """
        Run method of this Thread subclass (called in new thread through Thread.start())
        Loops until either:
            a) kill_switch.isSet() returns True
            b) _ShutdownQueue is retrieved from the scenario_path_queue
            c) shutdown() is called from the outside, triggering a) + b)

        Runs in:
            Only *this* background thread.
        """
        logger.info(f"Starting {self.name}(thread_id={self.native_id}) with preprocessor={self._preprocessor.name}")
        # Intentionally not using with statement, as this would lead to the executor waiting for running futures
        self._executor = ProcessPoolExecutor(max_workers=self._max_workers)

        def done_callback(future: Future[T_ScenarioPreprocessorResult]) -> None:
            self._put_future_result(future)

        while not self._kill_switch.is_set():
            logger.debug(f"Attempting to get scenario_path from _scenario_path_queue")
            scenario_path = self._scenario_path_queue.get()
            # _ShutdownQueue is a special sentinel value used to close the queue and shutdown this executor and thread
            if scenario_path is _ShutdownQueue and self._kill_switch.is_set():
                logger.debug(f"Got _ShutdownQueue from scenario_path_queue, kill_switch was set, attempting to shutdown")
                break
            logger.debug(f"Submitting scenario_path={scenario_path} to executor")
            _future = self._executor.submit(preprocess_scenario, scenario_path, self._preprocessor)
            _future.add_done_callback(done_callback)

        logger.debug(f"Shutting down because kill_switch was set")
        self.shutdown()

    def _put_future_result(self, future: Future[T_ScenarioPreprocessorResult]) -> None:
        """
        Will be called, when the future is cancelled or finishes running.
        Puts the result of finished futures in the preprocessing_results_queue.
        Handles and logs exception which occur during preprocessing.

        Runs in:
            A completely separate thread !!!
            See: https://stackoverflow.com/a/51883938

        Args:
            future (Future[T_ScenarioProcessorResults]): the cancelled or finished future
        """
        try:
            # Mark one retrieved scenario path as done
            self._scenario_path_queue.task_done()
            preprocessing_results = future.result()
            logger.debug(f"Trying to put preprocessing result {preprocessing_results} in preprocessing results queue")
            # This intentionally blocks if the queue has a maxsize, pausing preprocessing until a result is consumed by Consumer 2
            # A deadlock during shutdown is prevented here by clearing the queue
            self._preprocessing_result_queue.put(preprocessing_results)
            logger.debug(f"Put preprocessing result {preprocessing_results} in preprocessing results queue")
        except CancelledError as e:
            logger.error(f"Future cancelled with {e}", exc_info=e)
        except TimeoutError as e:
            logger.error(f"Future timed out with {e}", exc_info=e)
        except Exception as e:
            logger.error(f"Future raised exception {e}", exc_info=e)

    def request_preprocessing(self, scenario_paths: Sequence[Path]) -> None:
        """
        Runs in:
            Either the main thread or *this* background thread, if is_looping == True and there are no more remaining tasks.
            Only the main thread, if is_looping == False.

        Args:
            scenario_paths (Sequence[Path]): scenario paths which should be preprocessed
        """
        self._last_requested_scenario_paths = scenario_paths
        for path in scenario_paths:
            self._scenario_path_queue.put(path)
            self._remaining_task_counter.release()

    def get_preprocessing_result(self) -> Optional[T_ScenarioPreprocessorResult]:
        """
        Runs in:
            Only the main thread.

        Returns:
            next preprocessing result if any are available or None if there are no more remaining results
        """
        has_result = self._remaining_task_counter.acquire(blocking=False)
        if not has_result:
            if not self._is_looping:
                # Done
                return None
            # Loop
            self.request_preprocessing(self._last_requested_scenario_paths)
            return self.get_preprocessing_result()
        # Get available results
        result = self._preprocessing_result_queue.get()
        if result is _ShutdownQueue:
            return None
        return result

    def shutdown(self) -> None:
        """
        Shuts down all resources of this thread, such that the thread can be safely joined:
            * both Queue's
            * the ProcessPoolExecutor

        Achieves this by:
            1) Setting a kill switch Event
            2) Placing a _ShutdownQueue sentinel value into the scenario_path_queue, unblocking the (potentially blocked) get call
            3) This unblocks this thread and breaks the loop of the run() method
            4) Clearing the preprocessing_results_queue, this is necessary because Producer 2 could be blocked while trying to put a result into this queue
            5) Placing a _ShutdownQueue sentinel value into the preprocessing_results_queue, unblocking the (potentially blocked) get call
            6) This unblocks the main thread waiting for results
            7) Shutting down the ProcessPoolExecutor, cancels all remaining pending futures if cancel_futures is set

        Runs in:
            Either the main thread or *this* background thread.
        """
        # Immediately return if the kill_switch was already set, as another shutdown call must already be in progress
        if self._kill_switch.is_set():
            return

        logger.debug(f"Shutting down background {self.name}(thread_id={self.native_id}) with {self._remaining_task_counter._value} remaining tasks")
        logger.debug(f"Setting kill_switch")
        self._kill_switch.set()
        logger.debug(f"State of kill_switch={self._kill_switch.is_set()}")

        logger.debug(f"Putting _ShutdownQueue into scenario_path_queue")
        self._scenario_path_queue.put(_ShutdownQueue)

        logger.debug(f"Trying to clear preprocessing_results_queue")
        # This is somewhat questionable, as mutex is not part of the public API
        # Fine for now as we just try to clean up here
        # See: https://stackoverflow.com/a/18873213
        with self._preprocessing_result_queue.mutex:
            self._preprocessing_result_queue.queue.clear()
            self._preprocessing_result_queue.all_tasks_done.notify_all()
            self._preprocessing_result_queue.unfinished_tasks = 0
            logger.debug(f"Cleared preprocessing_results_queue")

        # Deadlock unless we release self._preprocessing_results_queue.mutex before doing this
        logger.debug(f"Putting _ShutdownQueue into preprocessing_results_queue")
        self._preprocessing_result_queue.put(_ShutdownQueue)
        # Increase task counter by 1 to assure that _ShutdownQueue will be retrieved
        self._remaining_task_counter.release()

        logger.debug(f"Shutting down ProcessPoolExecutor, cancel_futures={True}")
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info(f"Shut down background {self.name}(thread_id={self.native_id})")
