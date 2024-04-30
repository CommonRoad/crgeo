from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional, Sequence

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle

T_ScenarioPreprocessorInput = ScenarioBundle
T_ScenarioPreprocessorResult = Sequence[T_ScenarioPreprocessorInput]

logger = logging.getLogger(__name__)


class BaseScenarioPreprocessor(ABC, AutoReprMixin, StringResolverMixin):
    """
    Base class for preprocessing and filtering scenarios.
    Has two implementations: ScenarioPreprocessor and ScenarioFilter

    Uses magic/dunder methods to define a "domain-specific language" usable with
    the preprocessors.
    This DSL builds a directed-acyclic preprocessing graph which can be traversed
    with the child_preprocessors property.

    Example:

    .. code-block:: python

        inverted_second_filter = ~second_fiter
        chained_filter = first_filter >> inverted_second_filter
        chain_preprocessor_after_chained_filter = chained_filter >> preprocessor
        result: T_ScenarioPreprocessorResult = chain_preprocessor_after_chained_filter(scenario_bundle)

    More information:
        https://rszalski.github.io/magicmethods/
        https://en.wikipedia.org/wiki/Domain-specific_language
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:
        self._name = type(self).__name__ if name is None else name
        self._call_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        """
        Returns:
            The children of this preprocessor when viewed as a DAG.

        """
        ...

    @property
    @abstractmethod
    def results_factor(self) -> int:
        """
        Returns:
            How many preprocessed scenarios this preprocessor maximally returns
        """
        ...

    @property
    def call_count(self) -> int:
        return self._call_count

    def __call__(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        self._call_count += 1
        result = self._process(scenario_bundle)
        return result

    @abstractmethod
    def _process(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        ...

    def __rshift__(self, other: BaseScenarioPreprocessor) -> BaseScenarioPreprocessor:
        """
        Magic method to chain two BaseScenarioPreprocessor's in sequence.
        More information: https://rszalski.github.io/magicmethods/

        Example:

        .. code-block:: python

            chained_preprocessor_filter = preprocessor >> filter
            chained_filter = first_filter >> second_filter
            result = chained_filter(scenario_bundle)

        Args:
            other (BaseScenarioPreprocessor): the second preprocessor

        Returns:
            ChainedScenarioPreprocessor of self (first) and other (second)

        """
        return ChainedScenarioPreprocessor(
            first_scenario_processor=self,
            second_scenario_processor=other
        )

    def __or__(self, other: BaseScenarioPreprocessor) -> BaseScenarioPreprocessor:
        """
        The OR operation of two preprocessors applies each preprocessor to the
        input scenario bundle individually, returning a preprocessed scenario
        bundle for each preprocessor.

        The OR operation of more than two preprocessors functions in the same exact way,
        still returning a preprocessed scenario bundle for each preprocessor.

        All nesting of OR expressions is removed at the implementation level,
        i.e. ((pp1 | pp2) | pp3) == (pp1 | pp2 | pp3).
        This preprocessor just applies pp1, pp2, pp3 to the input scenario
        bundle s, returning three results pp1(s), pp2(s), pp3(s) in order.
        Evaluation happens in exactly this order.

        However, this is no guarantee that this preprocessor is not called from
        some other process during evaluation!

        Example:

        .. code-block:: python

            simple_multi_preprocessor = self_preprocessor | other_preprocessor
            multi_preprocessor = pp1 | pp2 | pp3 | pp4

        Args:
            other (ScenarioPreprocessor): the other preprocessor

        Returns:
            MultiScenarioPreprocessor of self and other

        """
        return MultiScenarioPreprocessor(self, other)

    def __mul__(self, count: int) -> BaseScenarioPreprocessor:
        """
        The MUL operation of a preprocessor with an integer applies the
        preprocessor N amount of times, which is equivalent to using __or__
        N amount of times.

        Example:

        .. code-block:: python

            multi_preprocessor = preprocessor | preprocessor | preprocessor
            same_multi_preprocessor = preprocessor * 3  # Equiv. to multi_preprocessor

        Args:
            count (int): how many times to apply the preprocessor

        Returns:
            MultiScenarioPreprocessor of self, count many times
        """
        preprocessors = [self] * count
        return MultiScenarioPreprocessor(*preprocessors)


class ChainedScenarioPreprocessor(BaseScenarioPreprocessor):
    """
    Chains two BaseScenarioPreprocessor's in sequence.
    """

    def __init__(
        self,
        first_scenario_processor: BaseScenarioPreprocessor,
        second_scenario_processor: BaseScenarioPreprocessor
    ) -> None:
        self._first_scenario_processor = first_scenario_processor
        self._second_scenario_processor = second_scenario_processor
        super(ChainedScenarioPreprocessor, self).__init__(name=f'({self._first_scenario_processor.name} >> {self._second_scenario_processor.name})')

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return [self._first_scenario_processor, self._second_scenario_processor]

    @property
    def results_factor(self) -> int:
        """
        Returns:
            at most as many results as the product of the results factors of both preprocessors
        """
        return self._first_scenario_processor.results_factor * self._second_scenario_processor.results_factor

    def _process(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        """
        First evaluates the first scenario preprocessor, then passes any of its
        outputs to the second scenario preprocessor.

        Args:
            scenario_bundle (ScenarioBundle): the scenario bundle which should be preprocessed

        Returns:
            list of the preprocessed scenario bundles which were returned by the
            second preprocessor after having being preprocessed by the first preprocessor
        """
        first_processor_results: T_ScenarioPreprocessorResult = self._first_scenario_processor(scenario_bundle)
        if not first_processor_results:
            return []

        second_processor_results = []
        for result_bundle in first_processor_results:
            if result_bundle is None:
                logger.debug(f"{result_bundle} of {self._first_scenario_processor.name} is None")
                continue
            second_processor_result = self._second_scenario_processor(result_bundle)
            if second_processor_result:
                second_processor_results.extend(second_processor_result)

        return second_processor_results


class MultiScenarioPreprocessor(BaseScenarioPreprocessor):

    def __init__(
        self,
        *scenario_preprocessors: BaseScenarioPreprocessor,
    ) -> None:
        """
        Un-nests all MultiScenarioPreprocessor's contained within scenario_preprocessors
        during initialization.
        As this is done every time, we don't have to use recursion to fully
        un-nest contained MultiScenarioPreprocessor's, as they must have already
        been un-nested during their own initialization.

        Args:
            *scenario_preprocessors (ScenarioPreprocessor): multiple scenario preprocessors applied to the input
        """
        self._scenario_preprocessors: List[BaseScenarioPreprocessor] = []
        for preprocessor in scenario_preprocessors:
            if isinstance(preprocessor, MultiScenarioPreprocessor):
                # Invariance: ParallelScenarioPreprocessor have already been fully un-nested during creation
                # Consequence: No recursion necessary to fully un-nest preprocessor into self
                nested_parallel_preprocessors = preprocessor._scenario_preprocessors
                self._scenario_preprocessors.extend(nested_parallel_preprocessors)
            else:  # Other preprocessors, cannot be un-nested
                self._scenario_preprocessors.append(preprocessor)

        names = [pp.name for pp in self._scenario_preprocessors]
        name = f"({' | '.join(names)})"
        super(MultiScenarioPreprocessor, self).__init__(name=name)

    @property
    def child_preprocessors(self) -> List[BaseScenarioPreprocessor]:
        return self._scenario_preprocessors

    @property
    def results_factor(self) -> int:
        """
        Returns:
            at most as many results as the sum of the results factor of each preprocessor
        """
        return sum([pp.results_factor for pp in self.child_preprocessors])

    def _process(
        self,
        scenario_bundle: ScenarioBundle
    ) -> T_ScenarioPreprocessorResult:
        """
        Evaluates multiple scenario preprocessors.

        Args:
            scenario_bundle (ScenarioBundle): scenario bundle to be preprocessed by multiple scenario preprocessors

        Returns:
            list of the preprocessed scenario bundles of each preprocessor
        """
        results = []
        for preprocessor in self._scenario_preprocessors:
            # We have to make a copy of the preprocessed scenario and preprocessed_planning_problem_set
            # As each individual scenario preprocessor might further modify this scenario
            # ScenarioBundle.__copy__ implements this
            scenario_bundle_copy = copy(scenario_bundle)
            result = preprocessor(scenario_bundle_copy)
            results.extend(result)

        return results
