import json
import logging
import random
import shutil
import string
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union, Callable, Iterable, TYPE_CHECKING

import torch

from crgeo.common.progress_reporter import ProgressReporter
from crgeo.dataset.commonroad_dataset import CommonRoadDataset, SaveFunction, LoadFunction
from crgeo.dataset.types import TransformCallable

log = logging.getLogger(__name__)

TMP_DIR_PREFIX = "__tmp_transform"

    

def dataset_transformation(
    dataset: CommonRoadDataset,
    transform: TransformCallable,
    output_directory: Optional[Path] = None,
    num_workers: int = 0,
) -> None:
    import numpy as np

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
                scenario_index = dataset.index_to_scenario_index(i)
                new_samples_per_old_sample.append(
                    _transformation_worker(
                        scenario_index=scenario_index,
                        sample_index=i,
                        sample_path=Path(path),
                        output_directory=output_directory,
                        transform=transform,
                        save_fn=dataset._save_fn,
                        load_fn=dataset._load_fn,
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
                            scenario_index=dataset.index_to_scenario_index(i),
                            sample_index=i,
                            sample_path=Path(path),
                            output_directory=output_directory,
                            transform=transform,
                            save_fn=dataset._save_fn,
                            load_fn=dataset._load_fn,
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

                    time.sleep(1.0)

                new_samples_per_old_sample = [result.get() for result in async_results]

        num_samples_per_scenario = np.empty(dataset._samples_cumsum.shape[0], dtype=int)
        i_start = 0
        for idx, i_end in enumerate(dataset._samples_cumsum):
            # idx is scenario_id
            new_samples = new_samples_per_old_sample[i_start:i_end]
            new_sample_idx = 0
            for j, samples in enumerate(new_samples):
                for sample in range(samples):
                    old_file_name = _transformed_file_name(original_sample_index=i_start + j, new_sample_index=sample)
                    new_file_name = CommonRoadDataset.processed_file_name(scenario_index=idx, sample_index=new_sample_idx)
                    new_sample_idx += 1
                    (output_directory / old_file_name).rename(output_directory / new_file_name)

            num_samples_per_scenario[idx] = sum(new_samples)
            i_start = i_end

        samples_cumsum = np.cumsum(num_samples_per_scenario)

        samples_file = output_directory / dataset._samples_file.name
        with samples_file.open("w", encoding="utf-8") as f:
            json.dump(samples_cumsum.tolist(), f)

        log.info("Dataset transformation completed, renaming temporary directory to final output directory")
        shutil.rmtree(final_output_directory, ignore_errors=True)
        output_directory.rename(final_output_directory)
        progress.update(len(processed_files) + 1)


def _transformation_worker(
    scenario_index: int,
    sample_index: int,
    sample_path: Path,
    output_directory: Path,
    transform: TransformCallable,
    save_fn: SaveFunction,
    load_fn: LoadFunction,
) -> int:
    sample_data = load_fn(sample_path, map_location=torch.device("cpu"))
    new_sample_index = 0
    for transformed_sample_data in transform(scenario_index, sample_index, sample_data):
        output_file = output_directory / _transformed_file_name(sample_index, new_sample_index)
        save_fn(transformed_sample_data, output_file)
        new_sample_index += 1
    return new_sample_index


def _transformed_file_name(original_sample_index: int, new_sample_index: int) -> str:
    return f"transform-{original_sample_index:04d}-{new_sample_index:04d}.pt"
