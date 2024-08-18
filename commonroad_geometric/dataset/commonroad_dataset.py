from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union
import warnings
import sys
import os.path as osp
import os
import random
from datetime import datetime

import torch
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from torch_geometric.data.dataset import Dataset, files_exist, _repr

from commonroad_geometric.common.io_extensions.scenario_file_format import ScenarioFileFormat
from commonroad_geometric.common.logging import BraceMessage as __
from commonroad_geometric.common.progress_reporter import NoOpProgressReporter, ProgressReporter
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor

logger = logging.getLogger(__name__)

T_IntermediateData = TypeVar("T_IntermediateData", bound=Union[CommonRoadData, CommonRoadDataTemporal])
T_Data = TypeVar("T_Data", bound=Union[CommonRoadData, CommonRoadDataTemporal])

SaveFunction = Callable[[Any, Path], None]
LoadFunction = Callable[[Path, Optional[Any]], Any]
T_PreSaveTransform = Callable[[Scenario, PlanningProblemSet], Iterable[T_IntermediateData]]
T_PreSaveFilter = Callable[[T_IntermediateData], bool]
T_PostLoadTransform = Callable[[T_IntermediateData], T_Data]


class CommonRoadDatasetMissingException(FileNotFoundError):
    ...


class CommonRoadDatasetMisconfigurationException(ValueError):
    ...


@dataclass
class CommonRoadDatasetConfig:
    root: Optional[Path] = None
    raw_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    overwrite_processed_dir: bool = False
    file_prefix: str = "crd"
    index_padding: str = "08"  # Pad each index to 8 digits, e.g. 1 to 00000001

    file_format: ScenarioFileFormat = ScenarioFileFormat.XML
    filter_scenario_paths: Callable[[list[Path]], list[Path]] = field(default=lambda f: f)
    scenario_preprocessor: BaseScenarioPreprocessor = field(default=IdentityPreprocessor())
    pre_processing_workers: int = 1
    scenario_cache_size: int = 8

    pre_transform: Optional[T_PreSaveTransform[T_IntermediateData]] = None
    max_samples_per_scenario: T_CountParam = Unlimited
    pre_filter: Optional[T_PreSaveFilter[T_IntermediateData]] = None
    pre_transform_workers: int = 1
    pre_transform_worker_init_fn: Optional[Callable[[], None]] = None
    pre_transform_progress: bool = True
    post_load_transform: Optional[T_PostLoadTransform[T_IntermediateData, T_Data]] = None

    save_fn: Optional[SaveFunction] = field(default=torch.save)
    load_fn: Optional[LoadFunction] = field(default=torch.load)
    load_device: torch.device = field(default=torch.device("cpu"))

    def __post_init__(self) -> None:
        # The order here matters!
        # Very explicitly make sure that we have:
        # 1) EITHER at least the root directory
        if self.root is not None:
            self.raw_dir = self.raw_dir or self.root / "raw"
            self.processed_dir = self.processed_dir or self.root / "processed"

        # If requested, overwrite the processed_dir before checking other conditions
        if self.overwrite_processed_dir and self.processed_dir is not None:
            logger.info(f"Deleting existing dataset from '{self.processed_dir}'")
            shutil.rmtree(self.processed_dir, ignore_errors=True)

        # 2) OR a fully processed directory BUT no raw directory
        if self.processed_dir is not None and self.processed_dir.exists():
            has_pt_files = any(path
                               for path in self.processed_dir.iterdir()
                               if path.stem.startswith(self.file_prefix) and path.suffix == ".pt")
            if has_pt_files:
                return

            if self.raw_dir is None:
                raise CommonRoadDatasetMissingException(f"Processed dataset in {self.processed_dir} does not contain "
                                                        f"any '.pt' files.")

        # 3) OR a raw directory and an empty processed directory
        if self.raw_dir is None or self.raw_dir.is_file() or not any(self.raw_dir.iterdir()):
            raise CommonRoadDatasetMissingException(f"Raw dataset in {self.raw_dir} is missing or directory is empty.")

        if self.processed_dir is None:
            raise CommonRoadDatasetMisconfigurationException(f"Need an empty directory to process into.")

        if self.processed_dir.is_file():
            raise CommonRoadDatasetMisconfigurationException(f"Need an empty directory to process into, got a file "
                                                             f"{self.processed_dir=} instead.")

        if self.processed_dir.exists() and any(self.processed_dir.iterdir()):
            raise CommonRoadDatasetMisconfigurationException(f"Need an empty directory to process into, got a non-empty"
                                                             f" directory {self.processed_dir=} instead.")


def create_processed_file_name(
    file_prefix: str,
    index_padding: str,
    scenario_index: int,
    sample_index: int,
    scenario_id: str,
) -> str:
    file_name = (f"{file_prefix}-"
                 f"scenario_{scenario_index:{index_padding}}-"
                 f"sample_{sample_index:{index_padding}}-"
                 f"id_{scenario_id}.pt")
    return file_name

@dataclass
class ProcessedFileParsingResult:
    processed_path: Path
    scenario_id: str = ''
    scenario_index: int = 0
    sample_index: int = 0

    def __post_init__(self):
        # Extracting parts from the filename (stem)
        parts = self.processed_path.stem.split('-')
        if len(parts) >= 3:
            prefix, scenario, sample = parts[0], parts[1], parts[2]
            scenario_id_parts = parts[3:]  # assuming all remaining parts are scenario_id parts
            self.scenario_id = "".join(scenario_id_parts).removeprefix("id_")

            # Extracting indexes
            _, str_scenario_index = scenario.split('_')
            _, str_sample_index = sample.split('_')

            self.scenario_index = int(str_scenario_index)
            self.sample_index = int(str_sample_index)

class CommonRoadDataset(Dataset, Generic[T_IntermediateData, T_Data]):

    def __init__(
        self,
        config: CommonRoadDatasetConfig,
        collector: BaseDatasetCollector
    ):
        self.config = config
        self.collector = collector
        if self.config.pre_transform is None:
            # Use default pre transform
            self.config.pre_transform = partial(
                self.collector.collect,
                max_samples=self.config.max_samples_per_scenario,
            )

        super().__init__(
            root=self.config.root,
            transform=self.config.post_load_transform,
            pre_transform=self.config.pre_transform,
            pre_filter=self.config.pre_filter
        )

        self._idx_to_processed_path = {}
        self._indices = []
        self.refresh_index_to_processed_path()

    @property
    def raw_dir(self) -> Optional[Path]:  # PyGeo uses str
        return self.config.raw_dir

    @property
    def processed_dir(self) -> Path:  # PyGeo uses str
        return self.config.processed_dir

    @property
    def raw_paths(self) -> list[Path]:  # PyGeo uses str
        if self.raw_dir is None:
            return []
        scenario_paths = [path for path in self.config.raw_dir.glob("**/*.xml")]
        scenario_paths = self.config.filter_scenario_paths(scenario_paths)
        return scenario_paths

    @property
    def processed_paths(self) -> list[Path]:  # PyGeo uses str
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        processed_paths = [
            path
            for path in self.processed_dir.iterdir()
            if path.is_file() and path.stem.startswith(self.config.file_prefix) and path.suffix == ".pt"
        ]
        return processed_paths

    @property
    def raw_file_names(self) -> list[str]:
        return [path.name for path in self.raw_paths]

    @property
    def processed_file_names(self) -> list[str]:
        return [path.name for path in self.processed_paths]

    def refresh_index_to_processed_path(self) -> None:
        r"""
        Refreshes indices and index to path dictionary from processed directory.
        """
        if self.processed_dir is None or not self.processed_dir.exists():
            return

        self._indices = []
        self._idx_to_processed_path = {}
        processed_paths = sorted(self.processed_dir.iterdir(), key=lambda p: p.name)
        index = 0
        for processed_path in processed_paths:
            if not processed_path.is_file():
                continue
            if not processed_path.stem.startswith(self.config.file_prefix):
                continue
            if not processed_path.suffix == ".pt":
                continue

            prefix, scenario, sample, *scenario_id_ = processed_path.stem.split('-')
            _, str_scenario_index = scenario.split('_')
            _, str_sample_index = sample.split('_')
            scenario_id = "".join(scenario_id_).removeprefix("id_")

            if not isinstance(str_scenario_index, str) or not isinstance(str_sample_index, str):
                continue

            if str_scenario_index.isdigit() and str_sample_index.isdigit():
                scenario_index = int(str_scenario_index)
                sample_index = int(str_sample_index)
                logger.debug(__(r"Adding {index=} for {scenario_id=} with {processed_path=}",
                                index=index, scenario_id=scenario_id, processed_path=processed_path))
                self._indices.append(index)
                self._idx_to_processed_path[index] = processed_path
                index += 1
        logger.info(f"Refreshing {len(self._indices)} processed file indices from {self.processed_dir}")

    def process(self) -> None:
        if self._indices:
            logger.info(f"Found {len(self._indices)} samples, skipping processing")
            return
        scenario_iterator = ScenarioIterator(
            directory=self.raw_dir,
            file_format=self.config.file_format,
            filter_scenario_paths=self.config.filter_scenario_paths,
            preprocessor=self.config.scenario_preprocessor,
            is_looping=False,
            seed=None,
            workers=self.config.pre_processing_workers,
            cache_size=self.config.scenario_cache_size
        )

        logger.info(f"Processing up to {scenario_iterator.max_result_scenarios} scenarios from "
                    f"{len(scenario_iterator.scenario_paths)} scenario paths")
        if self.config.pre_transform_progress:
            progress_reporter = ProgressReporter(
                name=f"{type(self).__name__}.process",
                total=scenario_iterator.max_result_scenarios,
                unit="scenario"
            )
        else:
            progress_reporter = NoOpProgressReporter()

        if self.config.pre_transform_workers <= 1:
            for scenario_index, scenario_bundle in enumerate(scenario_iterator):
                _pre_transform_worker(
                    scenario=scenario_bundle.preprocessed_scenario,
                    planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
                    pre_transform=self.config.pre_transform,
                    pre_filter=self.config.pre_filter,
                    save_function=self.config.save_fn,
                    processed_dir=self.processed_dir,
                    file_prefix=self.config.file_prefix,
                    index_padding=self.config.index_padding,
                    scenario_index=scenario_index
                )
                progress_reporter.update(scenario_index)
        else:
            # Run pre_transform function in a pool of worker processes
            with Pool(
                processes=self.config.pre_transform_workers,
                initializer=self.config.pre_transform_worker_init_fn,
            ) as pool:
                async_results = []
                for scenario_index, scenario_bundle in enumerate(scenario_iterator):
                    result = pool.apply_async(
                        _pre_transform_worker,
                        args=(),  # no args
                        kwds=dict(
                            scenario=scenario_bundle.preprocessed_scenario,
                            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
                            pre_transform=self.config.pre_transform,
                            pre_filter=self.config.pre_filter,
                            save_function=self.config.save_fn,
                            processed_dir=self.processed_dir,
                            file_prefix=self.config.file_prefix,
                            index_padding=self.config.index_padding,
                            scenario_index=scenario_index
                        )
                    )
                    progress_reporter.update(scenario_index)
                    async_results.append(result)

                logger.info(f"Submitted {len(async_results)} scenarios to the pre_transform workers")
                successful_results: list[list[Path]] = []
                # wait for and get the results
                for num_ready, result in enumerate(async_results):
                    result.wait()
                    if result.ready() and not result.successful():
                        # pre_transform function raised an exception, log the exception here
                        try:
                            result.get()
                        except Exception as e:
                            logger.error(f"Pre transform raised the following exception:", exc_info=e)
                    else:  # successful
                        successful_results.append(result.get())

                total_saved_data = sum(len(paths) for paths in successful_results)
                logger.info(f"Received {len(successful_results)} results from the pre_transform workers with "
                            f"{total_saved_data} total {type(T_IntermediateData)} being saved")

        progress_reporter.close()
        self.refresh_index_to_processed_path()

    def get(self, idx: int) -> T_IntermediateData:
        r"""
        Loads a T_IntermediateData instance from disk without applying a transform before returning it.
        If you want to load an instance and apply the transform, call Dataset.__getitem__.

        Args:
            idx (int): Index of the T_IntermediateData instance, e.g. 724 to load 'crd_00000724_some-scenario.pt'.

        Notes:
            Automatically re-maps missing indices.
            E.g. if 'crd_000000001_A.pt' and 'crd_000000003_C.pt' exist, but 'crd_000000002_B.pt' is missing because
            an exception occurred during processing, then idx=1 will return 'crd_000000001_A.pt' and idx=2 will return
            'crd_000000003_C.pt'.
            This is consistent with the behavior of len(dataset), which in this case would return 2.

        Returns:
            T_IntermediateData instance loaded from disk.
        """
        try:
            index = self._indices[idx]
        except IndexError:
            logger.debug(f"Failed to retrieve {idx=}")
            index = self._indices[-1] # TODO
        if idx != index:
            logger.debug(__(r"Skipping missing {idx=} instead returning data for {index=}", idx=idx, index=index))
        preprocessed_path = self._idx_to_processed_path[index]
        try:
            data = self.config.load_fn(preprocessed_path, map_location=self.config.load_device)
        except RuntimeError:
            logger.error(f"Failed to retrieve {idx=}")
            index = self._indices[-1] # TODO
            preprocessed_path = self._idx_to_processed_path[index]
            data = self.config.load_fn(preprocessed_path, map_location=self.config.load_device)

        return data[1] if isinstance(data, tuple) else data
    
    def index_to_scenario_index(self, idx: int) -> int:
        """
        Returns the scenario index corresponding to a given dataset index.

        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            int: The scenario index from which this sample was derived.
        """
        try:
            # Get the processed path from the index
            processed_path = self._idx_to_processed_path[self._indices[idx]]
        except IndexError:
            logger.error(f"Index {idx} is out of bounds.")
            raise ValueError(f"Index {idx} is out of the dataset range.")

        # Extract the scenario index part from the file name
        # The file name format is expected to be something like 'crd-scenario_00000001-sample_00000002-id_someid.pt'
        file_name = processed_path.stem
        parts = file_name.split('-')
        scenario_part = parts[1]  # 'scenario_00000001'
        scenario_index_str = scenario_part.split('_')[1]  # '00000001'

        # Convert the extracted string index to an integer
        return int(scenario_index_str)

    def split(
        self,
        size: Union[int, float],
        shuffle_before_split: bool = False,
    ) -> tuple[CommonRoadDataset[T_IntermediateData, T_Data], CommonRoadDataset[T_IntermediateData, T_Data]]:
        # Map scenarios to indices
        scenario_to_indices = {}
        for idx in range(len(self._indices)):
            scenario_idx = self.index_to_scenario_index(idx)
            if scenario_idx not in scenario_to_indices:
                scenario_to_indices[scenario_idx] = []
            scenario_to_indices[scenario_idx].append(idx)

        # Convert scenario indices to list for easier manipulation
        grouped_indices = list(scenario_to_indices.values())

        if shuffle_before_split:
            random.shuffle(grouped_indices)  # Shuffle the groups of indices

        # Calculate split index
        if isinstance(size, float):
            size = int(size * len(grouped_indices))  # Calculate size as a fraction of total groups

        # Split the grouped indices
        test_indices_groups = grouped_indices[:size]
        train_indices_groups = grouped_indices[size:]

        # Flatten the grouped indices back into flat lists
        test_indices = [idx for group in test_indices_groups for idx in group]
        train_indices = [idx for group in train_indices_groups for idx in group]

        assert len(train_indices) > 0, f"Training set split cannot be empty. Is your dataset large enough ({len(scenario_to_indices)} scenarios)?"
        assert len(test_indices) > 0, f"Test set split cannot be empty. Is your dataset large enough ({len(scenario_to_indices)} scenarios)?"

        test_dataset = self.index_select(torch.tensor(test_indices))
        train_dataset = self.index_select(torch.tensor(train_indices))

        return test_dataset, train_dataset

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return len(self._indices)
    
    @property
    def scenario_indices(self) -> set[int]:
        scenario_indices = set()
        for idx in range(len(self._indices)):
            scenario_indices.add(self.index_to_scenario_index(idx))
        return scenario_indices

    def _process(self):
        processed_dir = str(self.processed_dir)
        processed_paths = [str(p) for p in self.processed_paths]

        f = osp.join(processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                "The `pre_transform` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-processing technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        f = osp.join(processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, pass "
                "`force_reload=True` explicitly to reload the dataset.")

        force_reload = self.force_reload if hasattr(self, 'force_reload') else False

        if not force_reload and files_exist(processed_paths):
            return

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        os.makedirs(processed_dir, exist_ok=True)
        self.process()

        path = osp.join(processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)

    def get_meta_info(self) -> dict:
        """
        Retrieves meta information about the CommonRoadDataset.

        
        Returns:
            dict: A dictionary containing metadata about the dataset.
        """
        meta_info = {
            "dataset_size": len(self),
            "number_of_scenarios": len(self.scenario_indices),
            "raw_directory": str(self.raw_dir) if self.raw_dir else "Not specified",
            "processed_directory": str(self.processed_dir),
            "number_of_processed_files": len(self.processed_paths),
            "last_update_date": None
        }

        # Find the most recent update date from the processed files
        if self.processed_paths:
            latest_timestamp = max(os.path.getmtime(path) for path in self.processed_paths)
            meta_info["last_update_date"] = datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        return meta_info

def _pre_transform_worker(
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
    pre_transform: T_PreSaveTransform,
    pre_filter: Optional[T_PreSaveFilter],
    save_function: SaveFunction,
    processed_dir: Path,
    file_prefix: str,
    index_padding: str,
    scenario_index: int
) -> list[Path]:
    processed_paths = []
    for sample_index, data in pre_transform(scenario, planning_problem_set):
        if data is not None:
            if pre_filter is None or pre_filter(data):
                processed_file_name = create_processed_file_name(
                    file_prefix=file_prefix,
                    index_padding=index_padding,
                    scenario_index=scenario_index,
                    sample_index=sample_index,
                    scenario_id=str(scenario.scenario_id)
                )
                processed_path = processed_dir / processed_file_name
                save_function(data, processed_path)
                processed_paths.append(processed_path)
    return processed_paths
