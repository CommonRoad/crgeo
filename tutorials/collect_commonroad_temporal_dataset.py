from __future__ import annotations

import sys, os; sys.path.insert(0, os.getcwd())

import shutil
from typing import Iterable, cast

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from crgeo.common.types import T_CountParam, Unlimited
from crgeo.common.utils.datetime import get_timestamp_filename
from crgeo.common.utils.filesystem import load_dill, save_dill
from crgeo.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from crgeo.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from crgeo.dataset.commonroad_dataset import CommonRoadDataset
from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
from crgeo.learning.experiment import GeometricExperiment, GeometricExperimentConfig

DATA_COLLECTOR_CLS = ScenarioDatasetCollector
SCENARIO_DIR = 'data/highway_test'
DATASET_DIR = 'tutorials/output/dataset_t40'
EXPORT_FILENAME = 'experiment_config'
EXPORT_FILETYPE = 'pkl'


class TemporalGeometricExperiment:
    def __init__(self, config: GeometricExperimentConfig) -> None:
        self._config = config
        
        self._extractor_factory=TemporalTrafficExtractorFactory(
            traffic_extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    # render = False,
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50)
            )),
            options=TemporalTrafficExtractorOptions(collect_num_time_steps=40)
        )

        self._collector = config.data_collector_cls(
            extractor_factory=self._extractor_factory,
            simulation_options=self._config.simulation_options,
            scenario_preprocessors=config.preprocessors,
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
        max_samples: T_CountParam = Unlimited

    ) -> Iterable[CommonRoadDataTemporal]:
        """Extract traffic data and preprocess scenarios

        Args:
            scenario (Scenario): Scenario to be processed.
            planning_problem_set (PlanningProblemSet): Planning problem set.
            max_samples (int, optional): Max samples to be generated per scenario. Defaults to 1.

        Returns:
            Iterable[CommonRoadDataTemporal]

        """
        for time_step, data in self._collector.collect(
                scenario=scenario,
                planning_problem_set=planning_problem_set,
                max_samples=max_samples
            ):
            if data is not None:
                yield data


    def get_dataset(
        self,
        scenario_dir: str,
        dataset_dir: str,
        overwrite: bool,
        pre_transform_workers: int,
        max_scenarios: T_CountParam = Unlimited,
        cache_data: bool = False,
    ) -> CommonRoadDataset[CommonRoadDataTemporal, CommonRoadDataTemporal]:
        if overwrite:
            shutil.rmtree(dataset_dir, ignore_errors=True)

        commonroad_dataset = CommonRoadDataset[CommonRoadDataTemporal, CommonRoadDataTemporal](
            raw_dir=scenario_dir,
            processed_dir=dataset_dir,
            pre_transform=self.pre_transform_scenario,
            pre_transform_progress=True,
            pre_transform_workers=pre_transform_workers,
            max_scenarios=max_scenarios,
            cache_data=cache_data
        )

        return commonroad_dataset

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
        config = cast(GeometricExperimentConfig, load_dill(file_path)) if config is None else config
        experiment = GeometricExperiment(config)
        return experiment 


if __name__ == '__main__':
    shutil.rmtree(DATASET_DIR, ignore_errors=True)

    experiment_config = GeometricExperimentConfig(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50.0),
            )
        ),
        data_collector_cls=ScenarioDatasetCollector,
        preprocessors=[],
        postprocessors=[]
    )
    experiment = TemporalGeometricExperiment(experiment_config)

    #collect CommonRoadDataset, which contains collected Iterable[CommonRoadDataTemporal]
    dataset = experiment.get_dataset(
        scenario_dir=SCENARIO_DIR,
        dataset_dir=DATASET_DIR,
        overwrite=True,
        pre_transform_workers=4,
        max_scenarios=1,
        cache_data=True
    )

    print("Done exporting graph dataset")

    """
    dataset[0] is a CommonRoadDataTemporal instance from dataset, you can inspect its properties in DEBUG CONSOLE
    dataset[0][0] is a CommonRoadData instance, which corresponds to the first timestep in dataset[0]
    One example usage is to take each vehicle's trajectory in the graph and its node feature as a batch trajectories,
    the batch shape is [num_vehicle x time_length x feature_dim]
    """
    batch=dataset[0].get_node_features_temporal_sequence()
    
    