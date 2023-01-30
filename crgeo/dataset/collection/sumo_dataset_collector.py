from typing import Any, List, Mapping, Optional, Tuple, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from crgeo.common.types import T_CountParam, Unlimited
from crgeo.dataset.collection.base_dataset_collector import BaseDatasetCollector
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from crgeo.dataset.preprocessing.base_scenario_preprocessor import T_LikeBaseScenarioPreprocessor
from crgeo.simulation.base_simulation import BaseSimulation
from crgeo.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions



# TODO: Is this used??

class SumoDatasetCollector(BaseDatasetCollector[TrafficExtractor, TrafficExtractorOptions]):
    """
    Collects static dataset from SUMO simulation.
    """

    def __init__(
        self,
        extractor_factory: TrafficExtractorFactory,
        simulation_options: Optional[SumoSimulationOptions] = None,
        scenario_preprocessors: Optional[List[T_LikeBaseScenarioPreprocessor]] = None,
        scenario_filterers: Optional[List[BaseScenarioFilterer]] = None,
        sumo_simulation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super(SumoDatasetCollector, self).__init__(
            extractor_factory=extractor_factory,
            scenario_preprocessors=scenario_preprocessors,
            scenario_filterers=scenario_filterers
        )
        self._simulation_options = simulation_options
        self._sumo_simulation_kwargs = sumo_simulation_kwargs if sumo_simulation_kwargs is not None else {}
        
    def _setup_scenario(
        self, 
        scenario: Union[Scenario, str], 
        planning_problem_set: Optional[PlanningProblemSet] = None,
        max_samples_per_scenario: T_CountParam = Unlimited,
    ) -> Tuple[BaseSimulation, T_CountParam]:
        self._simulation = SumoSimulation(
            initial_scenario=scenario,
            options=self._simulation_options,
            **self._sumo_simulation_kwargs
        )
        self._simulation.start()
        self._time_step = 0
        return self._simulation, max_samples_per_scenario # SUMO has infinitely many samples, make at least as many as requested extractable

    def _exit_scenario(self) -> None:
        super()._exit_scenario()
        self._simulation.close()

    def _extract_from_timestep(self, time_step: int) -> CommonRoadData:
        if time_step != self._time_step:
            raise RuntimeError("This collector only supports extracting from successive time-steps")
        data = self._extractor.extract(TrafficExtractionParams(index=time_step))
        self._time_step += 1
        return data
