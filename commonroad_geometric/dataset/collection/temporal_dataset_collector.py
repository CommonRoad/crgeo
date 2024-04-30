from typing import Iterable, Optional

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.collection.base_dataset_collector import BaseDatasetCollector
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.extraction.base_extractor import T_BaseExtractionParams
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, \
    TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams


class TemporalDatasetCollector(
    BaseDatasetCollector[
        TemporalTrafficExtractor,
        TemporalTrafficExtractorOptions,
        TrafficExtractionParams,
        CommonRoadDataTemporal
    ]
):
    r"""
    DatasetCollector for CommonRoadDataTemporal.
    """

    # Need to override this to provide a reasonable default
    def collect(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        extraction_params: Optional[T_BaseExtractionParams] = None,
        max_samples: Optional[T_CountParam] = Unlimited,
    ) -> Iterable[tuple[int, CommonRoadDataTemporal]]:
        extraction_params = extraction_params or TrafficExtractionParams()
        return super().collect(
            scenario=scenario,
            extraction_params=extraction_params,
            max_samples=max_samples,
            planning_problem_set=planning_problem_set
        )

    def _collect(
        self,
        extractor: TemporalTrafficExtractor,
        extraction_params: TrafficExtractionParams,
        num_samples: T_CountParam
    ) -> Iterable[tuple[int, CommonRoadDataTemporal]]:
        for time_step, scenario in extractor.simulation(num_time_steps=int(num_samples)):
            data = extractor.extract(
                time_step=time_step,
                params=extraction_params
            )
            yield time_step, data
