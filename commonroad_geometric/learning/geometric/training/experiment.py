from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Type
from typing import Type, List, cast
import os
import os
import shutil
from functools import partial
import logging

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.common.utils.datetime import get_timestamp_filename
from commonroad_geometric.common.utils.filesystem import load_dill, load_pickle, save_dill, save_pickle
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorsPipeline
from commonroad_geometric.dataset.transformation.base_transformation import BaseDataTransformation
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions


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
    extractor_factory: BaseExtractorFactory
    data_collector_cls: Type[BaseDatasetCollector]
    simulation_options: BaseSimulationOptions = field(default_factory=BaseSimulationOptions) 
    preprocessors: T_ScenarioPreprocessorsPipeline = field(default_factory=list)
    filterers: List[BaseScenarioFilterer] = field(default_factory=list)
    postprocessors: List[T_LikeBaseDataPostprocessor] = field(default_factory=list)
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
        self._extractor_factory = config.extractor_factory
        self._collector = config.data_collector_cls(
            extractor_factory=self._extractor_factory,
            scenario_filterers=config.filterers,
            scenario_preprocessors=config.preprocessors,
            simulation_options=config.simulation_options
        )

    @property
    def config(self) -> GeometricExperimentConfig:
        return self._config
    
    def create_name(self) -> str:
        return f"{self._config.data_collector_cls.__name__}-{get_timestamp_filename()}"

    def pre_transform_scenario(
        self,
        scenario: Scenario, 
        planning_problem_set: PlanningProblemSet,
        max_samples: T_CountParam = Unlimited,
        progress: bool = True
    ) -> Iterable[CommonRoadData]:
        """Extract traffic data and preprocess scenarios

        Args:
            scenario (Scenario): Scenario to be processed.
            planning_problem_set (PlanningProblemSet): Planning problem set.
            max_samples (int, optional): Max samples to be generated per scenario. Defaults to 1.

        Returns:
            Iterable[CommonRoadData]

        Yields:
            Iterator[Iterable[CommonRoadData]]
        """

        if self._config.postprocessors:
            samples = [data for time_step, data in self._collector.collect(
                scenario,
                planning_problem_set=planning_problem_set,
                max_samples=max_samples,
                progress=progress
            ) if data is not None]
            if samples:
                for post_processor in self._config.postprocessors:
                    samples = post_processor(
                        samples=samples,
                        simulation=self._collector._simulation
                    )
            yield from samples
        else:
            for time_step, data in self._collector.collect(
                scenario,
                planning_problem_set=planning_problem_set,
                max_samples=max_samples,
                progress=progress
            ):
                if data is not None:
                    yield data

    def get_dataset(
        self,
        *,
        scenario_dir: str,
        dataset_dir: str,
        overwrite: bool,
        pre_transform_workers: int,
        max_scenarios: Optional[T_CountParam] = Unlimited,
        cache_data: bool = False,
        max_samples_per_scenario: Optional[T_CountParam] = Unlimited,
        collect_missing_samples: bool = True
    ) -> CommonRoadDataset:
        if overwrite:
            logger.info("Deleting existing dataset from '{dataset_dir}'")
            shutil.rmtree(dataset_dir, ignore_errors=True)

        if max_samples_per_scenario is None:
            max_samples_per_scenario = Unlimited
        if max_scenarios is None:
            max_scenarios = Unlimited
        pre_transform_scenario = partial(
            self.pre_transform_scenario,
            max_samples=max_samples_per_scenario
        ) if pre_transform_workers > 0 and collect_missing_samples else None

        commonroad_dataset = CommonRoadDataset(
            raw_dir=scenario_dir,
            processed_dir=dataset_dir,
            pre_transform=pre_transform_scenario,
            pre_transform_progress=True,
            pre_transform_workers=pre_transform_workers,
            max_scenarios=max_scenarios,
            cache_data=cache_data
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
    def _get_file_path(directory: str) -> str:
        return os.path.join(directory, EXPORT_FILENAME + '.' + EXPORT_FILETYPE)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        experiment_path = self._get_file_path(directory)
        save_dill(self._config, experiment_path)
        return experiment_path

    @classmethod
    def load(cls, file_path: str, config: GeometricExperimentConfig = None) -> GeometricExperiment:
        file_path = cls._get_file_path(file_path) if not file_path.endswith(EXPORT_FILETYPE) else file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        config = cast(GeometricExperimentConfig, load_dill(file_path)) if config is None else config
        experiment = GeometricExperiment(config)
        return experiment
