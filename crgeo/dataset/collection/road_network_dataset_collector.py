from typing import List, Optional, Sequence, Iterable, Tuple, Union, Type
import logging
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from crgeo.common.types import T_CountParam
from crgeo.dataset.collection.base_dataset_collector import BaseDatasetCollector
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.extraction.base_extractor import TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions
from crgeo.dataset.extraction.base_extractor_factory import BaseExtractorFactory
from crgeo.dataset.extraction.road_network.road_network_extractor import RoadNetworkExtractor, RoadNetworkExtractorOptions, RoadNetworkExtractionParams
from crgeo.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from crgeo.dataset.preprocessing.base_scenario_preprocessor import T_LikeBaseScenarioPreprocessor
from crgeo.simulation.base_simulation import BaseSimulation, Unlimited
from crgeo.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions

logger = logging.getLogger(__name__)

class RoadNetworkDatasetCollector(BaseDatasetCollector[RoadNetworkExtractor, RoadNetworkExtractorOptions]):
    def __init__(
        self,
        extractor_factory: BaseExtractorFactory[TypeVar_BaseExtractor, TypeVar_BaseExtractorOptions],
        scenario_preprocessors: Optional[List[T_LikeBaseScenarioPreprocessor]] = None,
        scenario_filterers: Optional[List[BaseScenarioFilterer]] = None,
    ) -> None:
        super(RoadNetworkDatasetCollector, self).__init__(
            extractor_factory=extractor_factory, 
            scenario_preprocessors=scenario_preprocessors,
            scenario_filterers=scenario_filterers
        )
        self._simulation_options = ScenarioSimulationOptions()
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
    
    def collect(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        max_samples: Union[int, Type[Unlimited]] = Unlimited,
        report_progress: bool = True,
    ) -> Iterable[CommonRoadData]:
        """Extracts graphs from each of the given CommonRoad scenarios.

        Args:
            scenario (Scenario):
                CommonRoad scenario.
            planning_problem_set (PlanningProblemSet, optional):
                CommonRoad planning problem set.
            max_samples (int or Unlimited, optional):
                Maximum number of samples to collect. Defaults to unlimited.
            report_progress (bool):
                Whether to report progress. Defaults to True.
        Returns:
            Iterable[List[CommonRoadData]]:
                A list of extracted and post-processed samples for each CommonRoad scenario.
        """
       
        # pre-process scenario
        if self._scenario_preprocessors is not None:
            for preprocessor, _ in self._scenario_preprocessors:
                scenario = preprocessor(scenario)
        
        simulation, _ = self._setup_scenario(scenario=scenario)

        try:
            self._extractor = self._extractor_factory.create(simulation) # Do we really need this? `simulation` parameter is unnecessary for road network extraction
        except Exception as e:
            print(f'Exception occured while creating extractor {str(e)}')
            return None

        for edge_index in range(len(self._extractor)):
            data = self._extractor.extract(RoadNetworkExtractionParams(index=edge_index))
            yield data
            