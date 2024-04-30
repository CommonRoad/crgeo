from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Type, cast

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.io_extensions.scenario_files import filter_max_scenarios
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import load_dill, save_dill
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset, CommonRoadDatasetConfig
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor
from commonroad_geometric.dataset.transformation.base_transformation import BaseDataTransformation
from commonroad_geometric.simulation.base_simulation import BaseSimulationOptions
from commonroad_geometric.simulation.simulation_factory import SimulationFactory

EXPORT_FILENAME = 'experiment_config'
EXPORT_FILETYPE = 'pkl'

logger = logging.getLogger(__name__)


@dataclass
class GeometricExperimentConfig:
    """
    traffic_extraction_options: Options for traffic extractor, which extract node and edge features for CommonRoadData
    data_collector_cls: Class of scenarioDatasetCollector, which utilizes traffic extractor for CommonRoadData collection
    simulation_options: Dataset is collected in simulation, chance to modify simulation options
    preprocessors: modify obstacles,lanelet graph or planning problem set in scenario before passing scenarios to data_collector_cls
    postprocessors: modify or compute extra attributes for CommonRoadData after data collection
    enable_anomaly_detection: True to enable torch.autograd.set_detect_anomaly
                            for better error logging for inplace operations that throw errors in automatic differentiation.
    """
    dataset_collector_cls: Type[BaseDatasetCollector]
    extractor_factory: BaseExtractorFactory
    simulation_options: BaseSimulationOptions
    scenario_preprocessor: BaseScenarioPreprocessor = field(default_factory=IdentityPreprocessor)
    transformations: List[BaseDataTransformation] = field(default_factory=list)
    enable_anomaly_detection = False


class GeometricExperiment:
    """
    GeometricExperiment save the experiment configs (e.g. the preprocessor, data_collector, traffic_extractor, post-processor)
    in experiment_config.pkl and can be loaded at test time to reproduce all the data processing at train time.

    Note: The functionality can be easily extended by adding other attributes (e.g. the distribution of training data)
    computed by e.g. post-processors to GeometricExperimentConfig

    """

    def __init__(self, config: GeometricExperimentConfig) -> None:
        self._config = config

    @property
    def config(self) -> GeometricExperimentConfig:
        return self._config

    def create_name(self) -> str:
        return f"{self._config.dataset_collector_cls.__name__}-{get_timestamp_filename()}"

    def get_dataset(
        self,
        *,
        scenario_dir: Path,
        dataset_dir: Path,
        overwrite: bool,
        pre_transform_workers: int,
        max_scenarios: Optional[T_CountParam] = Unlimited,
        max_samples_per_scenario: Optional[T_CountParam] = Unlimited,
    ) -> CommonRoadDataset:
        scenario_dir = Path(scenario_dir)
        dataset_dir = Path(dataset_dir)

        collector = self.config.dataset_collector_cls(
            extractor_factory=self.config.extractor_factory,
            simulation_factory=SimulationFactory(options=self.config.simulation_options),
        )
        commonroad_dataset = CommonRoadDataset(
            config=CommonRoadDatasetConfig(
                raw_dir=scenario_dir,
                processed_dir=dataset_dir,
                overwrite_processed_dir=overwrite,
                filter_scenario_paths=partial(filter_max_scenarios, max_scenarios=max_scenarios),
                scenario_preprocessor=self.config.scenario_preprocessor,
                pre_transform_workers=pre_transform_workers,
                max_samples_per_scenario=max_samples_per_scenario,
                pre_transform_progress=True,
            ),
            collector=collector
        )
        return commonroad_dataset

    def transform_dataset(self, dataset: CommonRoadDataset) -> CommonRoadDataset:
        for transformation in self.config.transformations:
            dataset = transformation.transform_dataset(dataset)
        return dataset

    def transform_data(self, data: CommonRoadData) -> CommonRoadData:
        for transformation in self.config.transformations:
            data = transformation.transform_data(data)
        return data

    @staticmethod
    def _get_file_path(directory: Path) -> Path:
        directory = Path(directory)
        return directory.joinpath(EXPORT_FILENAME + '.' + EXPORT_FILETYPE)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        experiment_path = self._get_file_path(directory)
        save_dill(self._config, experiment_path)
        return experiment_path

    @classmethod
    def load(cls, file_path: Path, config: GeometricExperimentConfig = None) -> GeometricExperiment:
        file_path = Path(file_path)
        file_path = cls._get_file_path(file_path) if not file_path.name.endswith(EXPORT_FILETYPE) else file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        config = cast(GeometricExperimentConfig, load_dill(file_path)) if config is None else config
        experiment = GeometricExperiment(config)
        return experiment
