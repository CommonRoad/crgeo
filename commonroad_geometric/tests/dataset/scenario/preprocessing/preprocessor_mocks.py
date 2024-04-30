import multiprocessing
from typing import Optional

import pytest
from commonroad.scenario.lanelet import Lanelet

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.filters import ScenarioFilter
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors import ScenarioPreprocessor


# Simple preprocessors for tests, implementing easier than mocking here
class RemoveLaneletsPreprocessor(ScenarioPreprocessor):
    def __init__(
        self,
        removed_lanelet_count: Optional[int] = None
    ) -> None:
        self.removed_lanelet_count = removed_lanelet_count
        super().__init__()

    def _process(
        self,
        scenario_bundle: ScenarioBundle,
    ) -> T_ScenarioPreprocessorResult:
        lanelets = scenario_bundle.preprocessed_scenario.lanelet_network.lanelets
        if self.removed_lanelet_count is not None:
            removed_lanelets = lanelets[:self.removed_lanelet_count]
        else:
            removed_lanelets = lanelets

        scenario_bundle.preprocessed_scenario.remove_lanelet(removed_lanelets)
        return [scenario_bundle]


class AddLaneletPreprocessor(ScenarioPreprocessor):
    def __init__(
        self,
        lanelet: Lanelet
    ) -> None:
        self.lanelet = lanelet
        super().__init__()

    def _process(
        self,
        scenario_bundle: ScenarioBundle,
    ) -> T_ScenarioPreprocessorResult:
        scenario_bundle.preprocessed_scenario.add_objects(self.lanelet)
        return [scenario_bundle]


class AcceptNFilter(ScenarioFilter):
    """
    Accepts n scenarios, then rejects every further scenario.
    """

    def __init__(self, n: int):
        self.counter = 0
        self.n = n
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        if self.counter < self.n:
            self.counter += 1
            return True
        return False


class SafeAcceptNFilter(ScenarioFilter):
    """
    Accepts n scenarios, then rejects every further scenario.
    Safe for use with multiprocessing.

    More information:
        https://stackoverflow.com/questions/28267972/python-multiprocessing-locks
        https://superfastpython.com/process-safe-counter/
    """

    def __init__(self, n: int):
        manager = multiprocessing.Manager()
        self.lock = manager.Lock()
        self.counter = manager.Value('i', 0)
        self.n = n
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        with self.lock:
            if self.counter.value < self.n:
                self.counter.value += 1
                return True
            return False


# Duplicates for testing
class _MinLaneletCountFilter(ScenarioFilter):
    """
    Rejects scenarios where the amount of lanelets in the lanelet network is more than the size threshold.
    """

    def __init__(self, min_size_threshold: int):
        self.min_size_threshold = min_size_threshold
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) >= self.min_size_threshold


class _MaxLaneletCountFilter(ScenarioFilter):
    """
    Rejects scenarios where the amount of lanelets in the lanelet network is less than the size threshold.
    """

    def __init__(self, max_size_threshold: int):
        self.max_size_threshold = max_size_threshold
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) <= self.max_size_threshold
