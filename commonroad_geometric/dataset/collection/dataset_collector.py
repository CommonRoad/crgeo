from typing import Iterable, Optional

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.base_extractor import T_BaseExtractionParams
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import T_LikeBaseDataPostprocessor
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import (TrafficExtractionParams,
                                                                               TrafficExtractor,
                                                                               TrafficExtractorOptions)
from commonroad_geometric.simulation.simulation_factory import SimulationFactory


class DatasetCollector(
    BaseDatasetCollector[TrafficExtractor, TrafficExtractorOptions, TrafficExtractionParams, CommonRoadData]
):
    r"""
    DatasetCollector for CommonRoadData.
    """

    def __init__(
        self,
        extractor_factory: BaseExtractorFactory[TrafficExtractor, TrafficExtractorOptions], 
        simulation_factory: SimulationFactory, progress: bool | ProgressReporter = True,
        deferred_postprocessors: Optional[list[T_LikeBaseDataPostprocessor]] = None
    ) -> None:
        self.deferred_postprocessors = deferred_postprocessors
        super().__init__(extractor_factory, simulation_factory, progress)

    def collect(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        extraction_params: Optional[T_BaseExtractionParams] = None,
        max_samples: Optional[T_CountParam] = Unlimited,
    ) -> Iterable[tuple[int, CommonRoadData]]:
        extraction_params = extraction_params or TrafficExtractionParams()
        return super().collect(
            scenario=scenario,
            extraction_params=extraction_params,
            max_samples=max_samples,
            planning_problem_set=planning_problem_set
        )

    def _collect(
        self,
        extractor: TrafficExtractor,
        extraction_params: TrafficExtractionParams,
        num_samples: T_CountParam
    ) -> Iterable[tuple[int, CommonRoadData]]:
        if not self.deferred_postprocessors:
            for time_step, scenario in extractor.simulation(num_time_steps=int(num_samples)):
                extraction_params.time_step = time_step
                data = extractor.extract(
                    time_step=time_step,
                    params=extraction_params
                )
                yield time_step, data
        else:
            timesteps = []
            samples = []
            for time_step, scenario in extractor.simulation(num_time_steps=int(num_samples)):
                extraction_params.time_step = time_step
                data = extractor.extract(
                    time_step=time_step,
                    params=extraction_params
                )

                timesteps.append(time_step)
                samples.append(data)
            
            for postprocessor in self.deferred_postprocessors:
                samples = postprocessor(
                    samples=samples,
                    simulation=extractor.simulation
                )

            for time_step, data in zip(timesteps, samples):
                yield time_step, data