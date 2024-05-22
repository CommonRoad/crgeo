import logging
from abc import abstractmethod
from typing import Generic, Iterable, Optional, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter, ProgressReporter
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.base_extractor import T_BaseExtractionParams, T_BaseExtractor, T_BaseExtractorOptions, T_ExtractionReturnType
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.simulation.simulation_factory import SimulationFactory

logger = logging.getLogger(__name__)


class BaseDatasetCollector(
    Generic[T_BaseExtractor, T_BaseExtractorOptions, T_BaseExtractionParams, T_ExtractionReturnType],
    AutoReprMixin
):
    r"""
    BaseDatasetCollector provides an interface for extracting and collecting PyTorch Geometric datasets from CommonRoad
    scenarios.

    This simplifies the process of building graph-based machine learning datasets.
    """

    def __init__(
        self,
        extractor_factory: BaseExtractorFactory[T_BaseExtractor, T_BaseExtractorOptions],
        simulation_factory: SimulationFactory,
        progress: Union[bool, ProgressReporter] = True,
    ) -> None:
        r"""
        Initializes the collector with the necessary factories to collect a sequence of T_ExtractionReturnType for a
        scenario.

        Args:
            extractor_factory (BaseExtractorFactory): Factory for extractor objects.
            simulation_factory (SimulationFactory): Factory for simulation objects.
            progress (Union[bool, ProgressReporter]): Whether to report progress. Defaults to True.
        """
        self._extractor_factory = extractor_factory
        self._simulation_factory = simulation_factory
        self._progress = progress
        logger.info(f"Initialized {type(self).__name__} with {extractor_factory=}, {simulation_factory=}")

    def collect(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        extraction_params: Optional[T_BaseExtractionParams] = None,
        max_samples: Optional[T_CountParam] = Unlimited,
    ) -> Iterable[tuple[int, T_ExtractionReturnType]]:
        r"""
        Extracts graphs from each of the given CommonRoad scenarios.

        Args:
            scenario (Scenario): CommonRoad scenario.
            planning_problem_set (PlanningProblemSet): Planning problem set for scenario.
            extraction_params (T_BaseExtractionParams): Parameters for extracting each sample.
            max_samples (Optional[T_CountParam]):  Maximum number of samples to collect. Defaults to unlimited.

        Returns:
            Iterable[tuple[int, T_ExtractionReturnType]]:
                An iterable of extracted and post-processed samples for each CommonRoad scenario.
        """
        if max_samples is None:
            max_samples = Unlimited
        assert max_samples is Unlimited or (isinstance(max_samples, int) and max_samples >= 0)
        if max_samples == 0:
            return

        simulation = self._simulation_factory(initial_scenario=scenario)
        simulation.start()
        # simulation.disable_step_rendering()

        available_samples = simulation.num_time_steps
        if available_samples is not Unlimited and available_samples == 0:
            return

        if max_samples is Unlimited:
            num_samples = available_samples
        else:
            num_samples = min(available_samples, max_samples) if available_samples is not Unlimited else max_samples
        assert num_samples is Unlimited or num_samples > 0

        progress_bar: BaseProgressReporter
        if self._progress:
            progress_bar = ProgressReporter(
                name=str(scenario.scenario_id),
                total=int(num_samples),
                parent_reporter=1
            )
        elif isinstance(self._progress, ProgressReporter):
            progress_bar = ProgressReporter(
                name=str(scenario.scenario_id),
                total=int(num_samples),
                parent_reporter=self._progress
            )
        else:
            progress_bar = NoOpProgressReporter()

        extractor = self._extractor_factory(simulation=simulation)
        try:
            for time_step, data in self._collect(
                extractor=extractor,
                extraction_params=extraction_params,
                num_samples=num_samples
            ):
                progress_bar.update(time_step)
                if time_step % 50 == 0:
                    progress_bar.display_memory_usage()
                yield time_step, data
        except Exception as e:
            logger.error(e, exc_info=True)
        finally:
            progress_bar.close()
            simulation.close()

    @abstractmethod
    def _collect(
        self,
        extractor: T_BaseExtractor,
        extraction_params: T_BaseExtractionParams,
        num_samples: T_CountParam
    ) -> Iterable[tuple[int, CommonRoadData]]:
        r"""
        Extracts graphs from each of the given CommonRoad scenarios.

        Args:
            extractor (T_BaseExtractor): Extractor, e.g. TrafficExtractor.
            num_samples (T_CountParam): Number of samples to collect.

        Returns:
            Iterable[tuple[int, T_ExtractionReturnType]]:
                An iterable of extracted and post-processed samples for each CommonRoad scenario.
        """
        ...
