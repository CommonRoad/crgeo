import json
import logging
import random
import shutil
import string
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union, Callable, Iterable
from tqdm import tqdm

import numpy as np
import torch
from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset, SaveFunction, LoadFunction, ProcessedFileParsingResult
from torch_geometric.data import Data, HeteroData

log = logging.getLogger(__name__)

AnyData = Union[Data, HeteroData, CommonRoadData]
TransformCallable = Callable[[int, int, AnyData], Iterable[AnyData]]

TMP_DIR_PREFIX = "__tmp_transform"


def dataset_transformation(
    dataset: CommonRoadDataset,
    transform: TransformCallable,
    output_directory: Optional[Path] = None,
    num_workers: int = 0,
) -> None:
    if output_directory is None:
        output_directory = Path(dataset.processed_dir)
    final_output_directory = output_directory

    # create temporary output directory
    random_name = "".join(random.sample(string.hexdigits, k=6))
    output_directory = output_directory.with_name(f"{TMP_DIR_PREFIX}_{output_directory.name}_{random_name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    log.info("Dataset transformation, temporary directory %s", output_directory)

    processed_files = dataset.processed_paths
    if len(processed_files) == 0:
        return

    with ProgressReporter(total=len(processed_files) + 1, unit="sample") as progress:
        if num_workers <= 1:
            new_samples_per_old_sample = []
            for i, path in enumerate(processed_files):
                new_samples_per_old_sample.append(
                    _transformation_worker(
                        sample_path=Path(path),
                        output_directory=output_directory,
                        transform=transform,
                        save_fn=dataset.config.save_fn,
                        load_fn=dataset.config.load_fn,
                    )
                )
                progress.update(i)
            progress.update(len(processed_files))

        else:
            pool = Pool(processes=min(num_workers, len(processed_files)))
            with pool:
                async_results = [
                    pool.apply_async(
                        func=_transformation_worker,
                        kwds=dict(
                            sample_path=Path(path),
                            output_directory=output_directory,
                            transform=transform,
                            save_fn=dataset.config.save_fn,
                            load_fn=dataset.config.load_fn,
                        ),
                    )
                    for i, path in enumerate(processed_files)
                ]
                # wait for results
                while True:
                    num_ready = 0
                    for result in async_results:
                        num_ready += int(result.ready())
                        if result.ready() and not result.successful():
                            # worker raised an exception, re-raise it here
                            result.get()

                    progress.update(num_ready)
                    if num_ready == len(async_results):
                        break

                new_samples_per_old_sample = [result.get() for result in async_results]

        log.info("Dataset transformation completed, renaming temporary directory to final output directory")
        shutil.rmtree(final_output_directory, ignore_errors=True)
        output_directory.rename(final_output_directory)
        progress.update(len(processed_files) + 1)


def _transformation_worker(
    sample_path: Path,
    output_directory: Path,
    transform: TransformCallable,
    save_fn: SaveFunction,
    load_fn: LoadFunction,
) -> int:
    sample_data = load_fn(sample_path, map_location=torch.device("cpu"))
    if isinstance(sample_data, tuple):
        sample_data = sample_data[-1]
    new_sample_index = 0
    parsing_result = ProcessedFileParsingResult(processed_path=sample_path)
    for transformed_sample_data in transform(parsing_result.scenario_index, parsing_result.sample_index, sample_data):
        output_file = output_directory / sample_path.name
        save_fn(transformed_sample_data, output_file)
        new_sample_index += 1
        break # TODO Allow multiple?
    tqdm.write(f"Transformed {parsing_result.scenario_id}/{parsing_result.sample_index}")
    return new_sample_index


def _transformed_file_name(original_sample_index: int, new_sample_index: int) -> str:
    return f"transform-{original_sample_index:04d}-{new_sample_index:04d}.pt"
