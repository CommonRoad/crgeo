from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import logging

from commonroad.scenario.scenario import Scenario
from crgeo.common.class_extensions.string_resolver_mixing import StringResolverMixin


logger = logging.getLogger(__name__)


class BaseScenarioFilterer(ABC, StringResolverMixin):
    """Base class for filtering scenarios."""

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = type(self).__name__ if name is None else name

    @property
    def name(self):
        return self._name

    def filter_scenario(
        self,
        scenario: Scenario,
        raise_exceptions: bool = False
    ) -> bool:

        if not raise_exceptions:
            value = self._filter_scenario(scenario)
        else:
            try:
                value = self._filter_scenario(scenario)
            except Exception as e:
                logger.error(e, exc_info=True)
                value = False

        if not value:
            logger.info(f"Filter {type(self).__name__} returned False for scenario {scenario.scenario_id}")
        return value

    @abstractmethod
    def _filter_scenario(self, scenario: Scenario) -> bool:
        """ """
