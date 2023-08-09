from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin

T_ScenarioPreprocessorCallable = Callable[[Scenario, Optional[PlanningProblemSet]], Tuple[Scenario, Optional[PlanningProblemSet]]]
T_LikeBaseScenarioPreprocessor = Union['BaseScenarioPreprocessor', T_ScenarioPreprocessorCallable]
T_ScenarioPreprocessorsInput = List[Union[Tuple[T_LikeBaseScenarioPreprocessor, int], T_LikeBaseScenarioPreprocessor]]
T_ScenarioPreprocessorsPipeline = List['BaseScenarioPreprocessor']

logger = logging.getLogger(__name__)


class BaseScenarioPreprocessor(ABC, AutoReprMixin, StringResolverMixin):

    """
    Base class for preprocessing scenario.
    As the first part of data preparation pipeline(preprocess -> dataset collector -> post-process), 
    preprocessors only deal with scenarios and planning problems, enabling operations on lanelet network, obstacles and planning problem setup
    and provide customized scenarios and planning problems for dataset collector.

    Current preprocessors can be easily extended by writing another child class of BaseScenarioPreprocessor and overwriting the abstractmethod "_process"
    """

    def __init__(self) -> None:
        self._name = self._get_name()
        self._call_count: int = 0

    def _get_name(self) -> str:
        return type(self).__name__

    @property
    def name(self) -> str:
        return self._name

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet],
        copy: bool = False
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        try:
            if copy:
                processed_scenario, processed_planning_problem_set = self._process(deepcopy(scenario), deepcopy(planning_problem_set))
            else:
                processed_scenario, processed_planning_problem_set = self._process(scenario, planning_problem_set)
        except Exception as e:
            logger.error(e, exc_info=True)
            return scenario, planning_problem_set

        self._call_count += 1
        return processed_scenario, processed_planning_problem_set

    @abstractmethod
    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet],
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        ...

    @staticmethod
    def cast(preprocessors: T_ScenarioPreprocessorsInput) -> Sequence[Tuple[BaseScenarioPreprocessor, int]]:
        casted_preprocessors = []
        for element in preprocessors:
            if isinstance(element, BaseScenarioPreprocessor):
                casted_preprocessor, repeat_count = element, 1
            elif isinstance(element, tuple):
                if isinstance(element[0], BaseScenarioPreprocessor):
                    casted_preprocessor, repeat_count = element # type: ignore # TODO
                else:
                    casted_preprocessor, repeat_count = FunctionalScenarioPreprocessor(element[0]), element[1]
            else:
                casted_preprocessor, repeat_count = FunctionalScenarioPreprocessor(element), 1
            casted_preprocessors.append((casted_preprocessor, repeat_count))
        return casted_preprocessors


class FunctionalScenarioPreprocessor(BaseScenarioPreprocessor):
    """Wrapper class for a stateless feature callable"""

    def __init__(self, scenario_preprocessor: T_ScenarioPreprocessorCallable) -> None:
        self._scenario_preprocessor = scenario_preprocessor
        super(FunctionalScenarioPreprocessor, self).__init__()

    def _get_name(self) -> str:
        return self._scenario_preprocessor.__name__

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        return self._scenario_preprocessor(scenario, planning_problem_set)
