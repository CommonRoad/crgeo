from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, List, Optional

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult

logger = logging.getLogger(__name__)


class ScenarioFilter(BaseScenarioPreprocessor):
    """
    Base class for filtering scenarios.

    Uses magic/dunder methods to define a DSL usable upon the filters.

    More information:
        https://rszalski.github.io/magicmethods/
        https://en.wikipedia.org/wiki/Domain-specific_language

    Example usage:

    .. code-block:: python

        or_filter = first_filter | second_filter
        result = chained_filter(scenario_bundle)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        is_inverted: bool = False
    ) -> None:
        super().__init__(name)
        self._is_inverted: bool = is_inverted

    @property
    def name(self) -> str:
        return f"{'~' if self._is_inverted else ''}{super().name}"

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        """
        Returns:
            ScenarioFilter usually does not have any children
        """
        return []

    @property
    def results_factor(self) -> int:
        """
        Returns:
            ScenarioFilter returns at most one scenario per scenario bundle
        """
        return 1

    def _process(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        # No public .filter method necessary as this class should be used with __call__
        # Renaming this to _filter for readability/clarity in implementation subclasses
        is_accepted: bool = self._filter(scenario_bundle)
        is_accepted = not is_accepted if self._is_inverted else is_accepted
        if is_accepted:
            return [scenario_bundle]

        logger.info(f"Filter {type(self).__name__} returned False for scenario {scenario_bundle.scenario_path.stem}")
        return []

    @abstractmethod
    def _filter(
        self,
        scenario_bundle: ScenarioBundle
    ) -> bool:
        ...

    def __invert__(self) -> ScenarioFilter:
        """
        Inverts a ScenarioFilter, if a filter accepts a scenario, the inverted filter rejects this scenario.
        Sole unary operator for ScenarioFilter.
        Side effect free through use of deepcopy, self is not modified.

        Returns:
            new inverted instance of ScenarioFilter

        """
        inverted_filter = deepcopy(self)
        inverted_filter._is_inverted = not self._is_inverted
        return inverted_filter

    # Binary operators
    def __and__(self, other: ScenarioFilter) -> ScenarioFilter:
        """
        The AND operation of two filters accepts a scenario when both filters accept the scenario.
        Hence chaining (>>, __rshift__) is semantically equivalent to the AND operation of two filters.

        Example usage:

        .. code-block:: python

            chain_preprocessor = first_filter >> second_filter # type: ChainedScenarioPreprocessor
            and_filter = first_filter & second_filter # type: ChainedScenarioPreprocessor

        Args:
            other (ScenarioFilter): the other filter

        Returns:
            ChainedScenarioPreprocessor of self and other
        """
        chained_filter = self.__rshift__(other)
        chained_filter._name = f'({self.name} & {other.name})'
        return chained_filter

    def __or__(self, other: ScenarioFilter) -> ScenarioFilter:
        """
        The OR operation of two filters accepts a scenario when EITHER or both filters accept the scenario.

        Example usage:

        .. code-block:: python

            or_filter = self_filter | other_filter

        Args:
            other (ScenarioFilter): the other filter

        Returns:
            _OrScenarioFilter of self and other
        """
        return _OrScenarioFilter(
            scenario_filter=self,
            other_scenario_filter=other
        )

    def __xor__(self, other: ScenarioFilter) -> ScenarioFilter:
        """
        The XOR operation of two filters accepting a scenario is equivalent to either but not both accepting the scenario.

        Example usage:

        .. code-block:: python

            or_filter = self_filter ^ other_filter

        Args:
            other (ScenarioFilter): the other filter

        Returns:
            _XorScenarioFilter of self and other
        """
        return _XorScenarioFilter(
            scenario_filter=self,
            other_scenario_filter=other
        )


class _OrScenarioFilter(ScenarioFilter):

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return [self._scenario_filter, self._other_scenario_filter]

    def __init__(
        self,
        scenario_filter: ScenarioFilter,
        other_scenario_filter: ScenarioFilter
    ) -> None:
        self._scenario_filter = scenario_filter
        self._other_scenario_filter = other_scenario_filter
        super().__init__(name=f'({self._scenario_filter.name} | {self._other_scenario_filter.name})')

    def _filter(
        self,
        scenario_bundle: ScenarioBundle
    ) -> bool:
        result: T_ScenarioPreprocessorResult = self._scenario_filter(scenario_bundle)
        other_result: T_ScenarioPreprocessorResult = self._other_scenario_filter(scenario_bundle)
        return result != [] or other_result != []


class _XorScenarioFilter(ScenarioFilter):

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return [self._scenario_filter, self._other_scenario_filter]

    def __init__(
        self,
        scenario_filter: ScenarioFilter,
        other_scenario_filter: ScenarioFilter
    ) -> None:
        self._scenario_filter = scenario_filter
        self._other_scenario_filter = other_scenario_filter
        super().__init__(name=f'({self._scenario_filter.name} ^ {self._other_scenario_filter.name})')

    def _filter(
        self,
        scenario_bundle: ScenarioBundle
    ) -> bool:
        result: T_ScenarioPreprocessorResult = self._scenario_filter(scenario_bundle)
        other_result: T_ScenarioPreprocessorResult = self._other_scenario_filter(scenario_bundle)
        return (result != [] and other_result == []) or (result == [] and other_result != [])


T_ScenarioFilterCallable = Callable[[ScenarioBundle], bool]


class FunctionalScenarioFilter(ScenarioFilter):
    """Wrapper class for a stateless scenario filter"""

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return []

    def __init__(self, filter_callable: T_ScenarioFilterCallable) -> None:
        self._scenario_filter = filter_callable
        super(FunctionalScenarioFilter, self).__init__(name=self._scenario_filter.__name__)

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return self._scenario_filter(scenario_bundle)
