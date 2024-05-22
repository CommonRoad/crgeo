from __future__ import annotations

import logging
from abc import ABCMeta
from copy import copy
from typing import Callable, List

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult

logger = logging.getLogger(__name__)


class ScenarioPreprocessor(BaseScenarioPreprocessor, metaclass=ABCMeta):
    """
    Base class for preprocessing scenarios and planning problems, enabling
    operations on lanelet network, obstacles and planning problem setup to
    provide customized scenarios and planning problems for downstream tasks.

    A new preprocessor is easily implemented by overwriting the abstractmethod
    "_process".

    If your preprocessor delegates to other preprocessors, indicate this by
    overwriting "child_preprocessors".

    If your preprocessor returns more than one preprocessing result per input
    scenario bundle, indicate the upper bound of the number of results by
    overwriting "results_factor".
    """

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        """
        Returns:
            ScenarioPreprocessor usually does not have any children
        """
        return []

    @property
    def results_factor(self) -> int:
        """
        Returns:
            ScenarioPreprocessor usually returns one preprocessed scenario per scenario bundle
        """
        return 1


T_ScenarioPreprocessorCallable = Callable[[ScenarioBundle], T_ScenarioPreprocessorResult]


class FunctionalScenarioPreprocessor(ScenarioPreprocessor):
    """Wrapper class for a stateless scenario preprocessor"""

    def __init__(self, scenario_processor: T_ScenarioPreprocessorCallable) -> None:
        self._scenario_processor = scenario_processor
        super(FunctionalScenarioPreprocessor, self).__init__(name=self._scenario_processor.__name__)

    def _process(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        return self._scenario_processor(scenario_bundle)
