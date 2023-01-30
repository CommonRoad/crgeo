import functools
import itertools
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

from commonroad.common.file_reader import CommonRoadFileReader

import numpy as np
import torch
import torch_geometric.data
Dataset = torch_geometric.data.Dataset

from crgeo.common.progress_reporter import NoOpProgressReporter, ProgressReporter
from crgeo.common.types import T_CountParam, Unlimited
from crgeo.dataset.types import T_IntermediateData, T_Data, T_PreFilter, T_PreTransform, T_Transform

logger = logging.getLogger(__name__)

SaveFunction = Callable[[Any, Path], None]
LoadFunction = Callable[[Path, Optional[Any]], Any]


class CommonRoadDatasetMissingException(FileNotFoundError):
    ...


class CommonRoadDataset(Dataset, Generic[T_IntermediateData, T_Data]):

    @staticmethod
    def is_processed(
        root: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> bool:            
        if root is not None:
            assert processed_dir is None, \
                "You have to specify either the root or the processed directory"
            processed_dir = Path(root) / "processed"
        else:
            assert processed_dir is not None, \
                "You have to specify either the root or the processed directory"
            processed_dir = Path(processed_dir)
        return (processed_dir / "samples.json").exists()

    @staticmethod
    def processed_file_name(scenario_index: int, sample_index: int) -> str:
        return f"data-{scenario_index:04d}-{sample_index:04d}.pt"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        pre_transform: Optional[T_PreTransform[T_IntermediateData]] = None,
        pre_filter: Optional[T_PreFilter[T_IntermediateData]] = None,
        pre_transform_workers: int = 0,
        pre_transform_worker_init_fn: Optional[Callable[[], None]] = None,
        pre_transform_progress: bool = True,
        transform: Optional[T_Transform[T_IntermediateData, T_Data]] = None,
        cache_data: bool = False,
        max_scenarios: T_CountParam = Unlimited,
        save_fn: Optional[SaveFunction] = None,
        load_fn: Optional[LoadFunction] = None,
    ):
        if root is not None:
            assert raw_dir is None and processed_dir is None, \
                "You have to specify either the root directory or the raw and processed directories"
            self._raw_dir = Path(root) / "raw"
            self._processed_dir = Path(root) / "processed"
        else:
            assert raw_dir is not None and processed_dir is not None, \
                "You have to specify either the root directory or the raw and processed directories"
            self._raw_dir = Path(raw_dir)
            self._processed_dir = Path(processed_dir)

        self._device = torch.device("cpu")
        self._save_fn = save_fn if save_fn is not None else torch.save
        self._load_fn = load_fn if load_fn is not None else torch.load
        self._pre_transform_workers = pre_transform_workers
        self.pre_transform_worker_init_fn = pre_transform_worker_init_fn
        self._pre_transform_progress = ProgressReporter if pre_transform_progress else NoOpProgressReporter
        self._max_scenarios = max_scenarios

        # _samples_cumsum stores the number of samples that were generated from all
        # raw files up to and including the current index
        self._samples_cumsum: Optional[np.ndarray] = None

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        if self._samples_cumsum is None:
            # process() was not called during initialization because the dataset is already processed
            # load the existing samples list from disk
            try:
                with self._samples_file.open(encoding="utf-8") as f:
                    samples_cumsum = json.load(f)
            except FileNotFoundError as e:
                raise CommonRoadDatasetMissingException(f"Dataset at '{self.raw_dir}' has not been preprocessed yet.")
            self._samples_cumsum = np.array(samples_cumsum, dtype=int)

        self._data_cache: Optional[Dict[int, T_IntermediateData]] = {} if cache_data else None
        # TODO self._data_cache retains more samples than necessary when index_select() is used

    @property
    def save_fn(self) -> SaveFunction:
        return self._save_fn

    @property
    def load_fn(self) -> SaveFunction:
        return self._load_fn

    @property
    def _samples(self) -> Iterable[int]:
        return itertools.chain(
            (self._samples_cumsum[0],),
            self._samples_cumsum[1:] - self._samples_cumsum[:-1],
        )

    @functools.cached_property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @functools.cached_property
    def processed_dir(self) -> Path:
        return self._processed_dir

    @functools.cached_property
    def raw_file_names(self) -> List[str]:
        filenames = sorted(str(file.relative_to(self._raw_dir)) for file in self._raw_dir.glob("**/*.xml"))
        if isinstance(self._max_scenarios, int):
            filenames = filenames[:self._max_scenarios]
        return filenames

    @functools.cached_property
    def processed_file_names(self) -> List[str]:
        return [
            self.processed_file_name(scenario_index, sample_index)
            for scenario_index, num_samples in enumerate(self._samples)
            for sample_index in range(num_samples)
        ]

    @functools.cached_property
    def _samples_file(self) -> Path:
        return self._processed_dir / "samples.json"

    def _process(self) -> None:
        if self._samples_file.exists() or self.pre_transform is None:
            # dataset has already been pre-processed
            return

        self._processed_dir.mkdir(exist_ok=True, parents=True)
        self.process()

    def process(self) -> None:
        assert self.pre_transform is not None, "pre_transform must be provided when processing the dataset"
        progress = self._pre_transform_progress(
            f"{type(self).__name__}.process",
            total=len(self.raw_paths),
            unit="scenario"
        )

        if len(self.raw_paths) == 0:
            raise ValueError(f"Dataset is empty (no filepaths founds at '{self.raw_dir}')")

        num_samples_per_scenario = []
        try:
            if self._pre_transform_workers <= 1:
                # run pre_transform function in this process
                total_sample_count: int = 0
                for scenario_index, scenario_file in enumerate(self.raw_paths):
                    num_samples_per_scenario.append(
                        _pre_transform_worker(
                            self._save_fn,
                            self.pre_transform,
                            self.pre_filter,
                            self._processed_dir,
                            scenario_index,
                            scenario_file,
                        )
                    )
                    total_sample_count += num_samples_per_scenario[-1]
                    progress.update(scenario_index)
                    progress.set_postfix_str(f"{scenario_file} ({num_samples_per_scenario[-1]}/{total_sample_count} samples)")
                progress.update(len(self.raw_paths))

            else:
                # run pre_transform function in a pool of worker processes
                num_workers = min(self._pre_transform_workers, len(self.raw_paths))
                with Pool(
                    processes=num_workers,
                    initializer=self.pre_transform_worker_init_fn,
                ) as pool:
                    async_results = [
                        pool.apply_async(
                            func=_pre_transform_worker,
                            args=(
                                self._save_fn,
                                self.pre_transform,
                                self.pre_filter,
                                self._processed_dir,
                                scenario_index,
                                scenario_file,
                            ),
                        )
                        for scenario_index, scenario_file in enumerate(self.raw_paths)
                    ]
                    # wait for the results
                    while True:
                        num_ready = 0
                        for result in async_results:
                            num_ready += int(result.ready())
                            if result.ready() and not result.successful():
                                # pre_transform function raised an exception, re-raise it here
                                result.get()

                        progress.update(num_ready)
                        if num_ready == len(async_results):
                            break
                        time.sleep(0.1)

                    num_samples_per_scenario = [ r.get() for r in async_results ]

        finally:
            if len(num_samples_per_scenario) > 0:
                self._samples_cumsum = np.cumsum(num_samples_per_scenario, dtype=int)
                with self._samples_file.open("w", encoding="utf-8") as f:
                    json.dump(self._samples_cumsum.tolist(), f)
            progress.close()

    def len(self) -> int:
        # size of the complete dataset without considering the selected subset
        return int(self._samples_cumsum[-1]) if len(self._samples_cumsum) > 0 else 0

    def index_to_scenario_index(self, idx: int) -> int:
        return int(np.searchsorted(self._samples_cumsum, idx + 1))

    def get(self, idx: int) -> T_IntermediateData:
        if self._data_cache is not None and idx in self._data_cache:
            return self._data_cache[idx]

        scenario_index = self.index_to_scenario_index(idx)
        if scenario_index == 0:
            sample_index = idx
        else:
            sample_index = idx - self._samples_cumsum[scenario_index - 1]
        path = self._processed_dir / self.processed_file_name(scenario_index, sample_index)
        data = self._load_fn(path, map_location=self._device)
        if self._data_cache is not None:
            self._data_cache[idx] = data
        return data

    def iter_raw_and_processed_files(self) -> Iterable[Tuple[Path, Path]]:
        for scenario_index, num_samples in enumerate(self._samples):
            raw_path = Path(self.raw_paths[scenario_index])
            for sample_index in range(num_samples):
                processed_path = self._processed_dir / self.processed_file_name(scenario_index, sample_index)
                yield raw_path, processed_path

    def split(
        self,
        size: Union[int, float],
        shuffle_before_split: bool = False,
    ) -> Tuple["CommonRoadDataset[T_IntermediateData, T_Data]", "CommonRoadDataset[T_IntermediateData, T_Data]"]:
        if isinstance(size, float):
            size = int(size * len(self))

        if shuffle_before_split:
            indices = torch.randperm(len(self), dtype=torch.long, device="cpu")
        else:
            indices = torch.arange(len(self), dtype=torch.long, device="cpu")

        return self.index_select(indices[:size]), self.index_select(indices[size:])


def _pre_transform_worker(
    save_function: SaveFunction,
    pre_transform: T_PreTransform,
    pre_filter: Optional[T_PreFilter],
    processed_dir: Path,
    scenario_index: int,
    scenario_file: str,
) -> int:
    try:
        # TODO: Use pickle (ScenarioIterator)
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    except Exception as e:
        logger.error(f"Failed to load scenario from '{scenario_file}'")
        logger.error(e, exc_info=True)
        return 0
    sample_index = -1
    for data in pre_transform(scenario, planning_problem_set):
        if data is not None:
            if pre_filter is None or pre_filter(data):
                sample_index += 1
            path = processed_dir / CommonRoadDataset.processed_file_name(scenario_index, sample_index)
            save_function(data, path)
    return sample_index + 1
