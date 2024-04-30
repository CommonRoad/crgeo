import asyncio
import concurrent.futures
import time
from asyncio import Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_geometric.common.io_extensions.scenario_files import ScenarioFileFormat, find_scenario_paths
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle


# sync
def read_scenario(scenario_path) -> ScenarioBundle:
    file_reader = CommonRoadFileReader(filename=str(scenario_path))
    scenario, planning_problem_set = file_reader.open()
    return ScenarioBundle(
        scenario_path=scenario_path,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )


def collect_scenario_bundles(_scenario_paths) -> List[ScenarioBundle]:
    _scenario_bundles = [read_scenario(scenario_path) for scenario_path in _scenario_paths]
    return _scenario_bundles


# ThreadPoolExecutor
def thread_pool_map_scenario_bundles(_scenario_paths, num_workers) -> List[ScenarioBundle]:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_scenario_bundle = executor.map(read_scenario, _scenario_paths)
        return list(future_to_scenario_bundle)


def thread_pool_submit_scenario_bundles(_scenario_paths, num_workers) -> List[ScenarioBundle]:
    _scenario_bundles = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_scenario_bundle = [executor.submit(read_scenario, path) for path in _scenario_paths]
        for future in concurrent.futures.as_completed(future_to_scenario_bundle):
            scenario_bundle = future.result()
            _scenario_bundles.append(scenario_bundle)
    return _scenario_bundles


# ProcessPoolExecutor
def process_pool_submit_scenario_bundles(_scenario_paths, num_workers) -> List[ScenarioBundle]:
    _scenario_bundles = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_scenario_bundle = [executor.submit(read_scenario, path) for path in _scenario_paths]
        for future in concurrent.futures.as_completed(future_to_scenario_bundle):
            scenario_bundle = future.result()
            _scenario_bundles.append(scenario_bundle)
    return _scenario_bundles


# asyncio
async def read_scenario_async(scenario_path) -> ScenarioBundle:
    file_reader = CommonRoadFileReader(filename=str(scenario_path))
    scenario, planning_problem_set = file_reader.open()
    return ScenarioBundle(
        scenario_path=scenario_path,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )


async def read_scenario_async_to_thread(scenario_path) -> ScenarioBundle:
    file_reader = CommonRoadFileReader(filename=str(scenario_path))
    coroutine = asyncio.to_thread(file_reader.open)
    scenario, planning_problem_set = await coroutine
    return ScenarioBundle(
        scenario_path=scenario_path,
        input_scenario=scenario,
        input_planning_problem_set=planning_problem_set
    )


# asyncio.gather
async def asyncio_gather_scenario_bundles(scenario_paths) -> List[ScenarioBundle]:
    _scenario_bundles = await asyncio.gather(
        *[read_scenario_async(scenario_path) for scenario_path in scenario_paths]
    )
    return _scenario_bundles


# asyncio.Queue
async def queue_worker(queue: Queue, return_queue: Queue):
    while True:
        scenario_path = await queue.get()
        scenario_bundle = await read_scenario_async_to_thread(scenario_path)
        await return_queue.put(scenario_bundle)
        queue.task_done()


async def asyncio_queue_scenario_bundles(scenario_paths, num_workers) -> List[ScenarioBundle]:
    queue = asyncio.Queue()
    for path in scenario_paths:
        queue.put_nowait(path)

    return_queue = asyncio.Queue()
    tasks = []
    for i in range(num_workers):
        task = asyncio.create_task(queue_worker(queue, return_queue))
        tasks.append(task)

    await queue.join()
    _scenario_bundles = []
    while not return_queue.empty():
        scenario_bundle = await return_queue.get()
        _scenario_bundles.append(scenario_bundle)

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)

    return _scenario_bundles


# asyncio.to_thread
async def asyncio_to_thread_scenario_bundles(scenario_paths):
    _scenario_bundles = await asyncio.gather(
        *[read_scenario_async_to_thread(path) for path in scenario_paths]
    )
    return _scenario_bundles


if __name__ == '__main__':
    # This file is the decision basis for using ProcessPoolExecutor for ScenarioIterator
    # https://superfastpython.com/multithreaded-file-loading/
    # https://superfastpython.com/concurrency-file-io/

    scenario_dir = Path('data', 'highd-sample')
    _scenario_paths = find_scenario_paths(scenario_dir, file_format=ScenarioFileFormat.XML)

    print(f"Using FileFormat={ScenarioFileFormat.XML}")

    # sync
    total_time = time.time_ns()
    scenario_bundles = collect_scenario_bundles(_scenario_paths)
    total_time = time.time_ns() - total_time
    print(f"synchronous code took: {total_time * 1e-9}")

    # ProcessPoolExecutor
    workers = [2, 4, 8]
    for _num_workers in workers:
        total_time = time.time_ns()
        process_pool_submit_scenario_bundles(_scenario_paths, _num_workers)
        total_time = time.time_ns() - total_time
        print(f"ProcessPoolExecutor.submit with num_workers={_num_workers} took: {total_time * 1e-9}")

    # Can spawn much more threads/tasks than processes
    workers += [16, 32]

    # ThreadPoolExecutor
    for _num_workers in workers:
        total_time = time.time_ns()
        thread_pool_map_scenario_bundles(_scenario_paths, _num_workers)
        total_time = time.time_ns() - total_time
        print(f"ThreadPoolExecutor.map with num_workers={_num_workers} took: {total_time * 1e-9}")

    for _num_workers in workers:
        total_time = time.time_ns()
        thread_pool_submit_scenario_bundles(_scenario_paths, _num_workers)
        total_time = time.time_ns() - total_time
        print(f"ThreadPoolExecutor.submit with num_workers={_num_workers} took: {total_time * 1e-9}")

    # asyncio
    total_time = time.time_ns()
    asyncio.run(asyncio_gather_scenario_bundles(_scenario_paths))
    total_time = time.time_ns() - total_time
    print(f"asyncio.gather took: {total_time * 1e-9}")

    total_time = time.time_ns()
    asyncio.run(asyncio_to_thread_scenario_bundles(_scenario_paths))
    total_time = time.time_ns() - total_time
    print(f"asyncio.gather with asyncio.to_thread took: {total_time * 1e-9}")

    for _num_workers in workers:
        total_time = time.time_ns()
        asyncio.run(asyncio_queue_scenario_bundles(_scenario_paths, _num_workers))
        total_time = time.time_ns() - total_time
        print(f"asyncio.Queue with num_workers={_num_workers} took: {total_time * 1e-9}")
