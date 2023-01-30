from typing import List, Optional, Tuple, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from crgeo.common.types import T_CountParam
from crgeo.dataset.collection.base_dataset_collector import BaseDatasetCollector
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from crgeo.dataset.preprocessing.base_scenario_preprocessor import T_LikeBaseScenarioPreprocessor
from crgeo.simulation.base_simulation import BaseSimulation, Unlimited
from crgeo.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions


class ScenarioDatasetCollector(BaseDatasetCollector[TrafficExtractor, TrafficExtractorOptions]):
    """
    ScenarioDatasetCollector provides a framework for extracting and collecting PyTorch Geometric datasets from
    CommonRoad scenarios.

    This simplifies the process of building graph-based machine learning datasets.
    """
    def __init__(
        self,
        extractor_factory: BaseExtractorFactory[TrafficExtractor, TrafficExtractorOptions],
        simulation_options: Optional[ScenarioSimulationOptions] = None,
        scenario_preprocessors: Optional[List[T_LikeBaseScenarioPreprocessor]] = None,
        scenario_filterers: Optional[List[BaseScenarioFilterer]] = None,
    ) -> None:
        super(ScenarioDatasetCollector, self).__init__(
            extractor_factory=extractor_factory,
            scenario_preprocessors=scenario_preprocessors,
            scenario_filterers=scenario_filterers
        )
        self._simulation_options = simulation_options or ScenarioSimulationOptions()
        self._simulation_options.backup_current_scenario = False
        self._simulation_options.backup_initial_scenario = False

    def _setup_scenario(
        self, scenario: Union[Scenario, str],
        planning_problem_set: Optional[PlanningProblemSet] = None,
        max_samples_per_scenario: T_CountParam = Unlimited,
    ) -> Tuple[BaseSimulation, T_CountParam]:
        self._simulation = ScenarioSimulation(
            initial_scenario=scenario,
            options=self._simulation_options
        )
        self._simulation.start()
        return self._simulation, self._simulation.num_time_steps

    def _exit_scenario(self) -> None:
        super()._exit_scenario()
        self._simulation.close()

    def _extract_from_timestep(self, time_step: int) -> CommonRoadData:
        if time_step < self._simulation.initial_time_step or \
                self._simulation.final_time_step is not Unlimited and time_step > self._simulation.final_time_step:
            raise RuntimeError("This collector only supports extracting from precomputed time-steps in the scenario")
        # Do this instead of calling step() function as we can jump through the precomputed ScenarioSimulation
        # self._simulation.current_time_step will be used within TrafficExtractor
        self._simulation.current_time_step = time_step
        data = self._extractor.extract(TrafficExtractionParams(index=time_step))
        return data
