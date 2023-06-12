import logging
from abc import abstractmethod
from typing import Generic, Iterable, List, Optional, Tuple, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.progress_reporter import BaseProgressReporter, NoOpProgressReporter, ProgressReporter
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.base_extractor import TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions
from commonroad_geometric.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import T_LikeBaseScenarioPreprocessor
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation

logger = logging.getLogger(__name__)


class BaseDatasetCollector(Generic[TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions], AutoReprMixin):
    def __init__(
        self,
        extractor_factory: BaseExtractorFactory[TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions],
        scenario_preprocessors: Optional[List[T_LikeBaseScenarioPreprocessor]] = None,
        scenario_filterers: Optional[List[BaseScenarioFilterer]] = None,
    ) -> None:
        self._extractor_factory = extractor_factory
        self._extractor: TypeVar_BaseExtractor
        self._scenario_preprocessors = scenario_preprocessors if scenario_preprocessors is not None else []
        self._scenario_filterers = scenario_filterers if scenario_filterers is not None else []

        logger.info(f"Initialized {type(self).__name__} with extractor factory of type '{type(extractor_factory).__name__}'")

    def collect(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        max_samples: Optional[T_CountParam] = Unlimited,
        progress: Union[bool, ProgressReporter] = True,
    ) -> Iterable[Tuple[int, CommonRoadData]]:
        """Extracts graphs from each of the given CommonRoad scenarios.

        Args:
            scenario (Scenario):
                CommonRoad scenario.
            planning_problem_set (PlanningProblemSet, optional):
                CommonRoad planning problem set.
            max_samples (int or Unlimited, optional):
                progress number of samples to collect. Defaults to unlimited.
            progress (Union[bool, ProgressReporter]):
                Whether to report progress. Defaults to True.
        Returns:
            Iterable[List[CommonRoadData]]:
                A list of extracted and post-processed samples for each CommonRoad scenario.
        """
        if max_samples is None:
            max_samples = Unlimited
        assert max_samples is Unlimited or (isinstance(max_samples, int) and max_samples >= 0)
        if max_samples == 0:
            return

        # TODO: https://gitlab.lrz.de/cps/commonroad-geometric/-/issues/244
        if self._scenario_filterers is not None:
            for filterer in self._scenario_filterers:
                if not filterer.filter_scenario(scenario):
                    return
        if self._scenario_preprocessors is not None:
            for preprocessor in self._scenario_preprocessors:
                scenario, planning_problem_set = preprocessor(scenario, planning_problem_set)

        try:
            simulation, available_samples = self._setup_scenario(
                scenario=scenario,
                planning_problem_set=planning_problem_set,
                max_samples_per_scenario=max_samples
            )
        except Exception as e:
            logger.error(e, exc_info=True)
            return

        simulation.disable_step_rendering()
        self._extractor = self._extractor_factory.create(simulation)
        if available_samples is not Unlimited and available_samples == 0:
            return

        if max_samples is Unlimited:
            num_samples = available_samples
        else:
            num_samples = min(available_samples, max_samples) if available_samples is not Unlimited else max_samples
        assert num_samples is Unlimited or num_samples > 0

        progress_bar: BaseProgressReporter
        if progress == True:
            progress_bar = ProgressReporter(
                name=str(scenario.scenario_id),
                total=int(num_samples),
                parent_reporter=1
            )
        elif isinstance(progress, ProgressReporter):
            progress_bar = ProgressReporter(
                name=str(scenario.scenario_id),
                total=int(num_samples),
                parent_reporter=progress
            )
        else:
            progress_bar = NoOpProgressReporter()

        try:
            for time_step, scenario in simulation(num_time_steps=int(num_samples)):
                data = self._extract_from_timestep(time_step)
                if data is not None and simulation.renderers:
                    simulation.render(render_params=RenderParams(
                        data=data
                    ))
                progress_bar.update(time_step)
                if time_step % 50 == 0:
                    progress_bar.display_memory_usage()
                yield time_step, data
        except Exception as e:
            logger.error(e, exc_info=True)
        finally:
            progress_bar.close()
            self._exit_scenario()

    @abstractmethod
    def _setup_scenario(
        self,
        scenario: Union[Scenario, str],
        planning_problem_set: Optional[PlanningProblemSet] = None,
        max_samples_per_scenario: T_CountParam = Unlimited
    ) -> Tuple[BaseSimulation, T_CountParam]:
        """
        Performs setup for scenario.

        Args:
            scenario (Union[Scenario, str]): Input scenario.
            max_samples_per_scenario (int or Unlimited): Maximum number of samples per scenario.

        Returns:
            tuple (BaseSimulation, number of samples to be collected).
        """

    def _exit_scenario(self) -> None:
        pass

    @abstractmethod
    def _extract_from_timestep(self, time_step: int) -> CommonRoadData:
        """
        Extracts data from the specified time-step.
        """
