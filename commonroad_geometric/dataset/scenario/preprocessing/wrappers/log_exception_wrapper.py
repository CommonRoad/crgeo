import logging
from typing import List

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult

logger = logging.getLogger(__name__)


class LogExceptionWrapper(BaseScenarioPreprocessor):
    """
    Wrapper class which suppresses and logs exceptions of the wrapped preprocessor.
    """

    def __init__(
        self,
        wrapped_preprocessor: BaseScenarioPreprocessor
    ) -> None:
        self.wrapped_preprocessor = wrapped_preprocessor
        logger.warning(f"Wrapping {self.wrapped_preprocessor.name} with {type(self).name}, suppressing exceptions!")
        super(LogExceptionWrapper, self).__init__(name=f"{type(self).name}({self.wrapped_preprocessor.name})")

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return self.wrapped_preprocessor.child_preprocessors

    @property
    def results_factor(self) -> int:
        return self.wrapped_preprocessor.results_factor

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        try:
            result = self.wrapped_preprocessor._process(scenario_bundle)
            return result
        except Exception as e:
            logger.error(e, exc_info=True)
            return []
