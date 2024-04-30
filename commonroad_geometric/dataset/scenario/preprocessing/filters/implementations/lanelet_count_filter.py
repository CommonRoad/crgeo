from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class MinLaneletCountFilter(ScenarioFilter):
    """
    Rejects scenarios where the amount of lanelets in the lanelet network is more than the size threshold.
    """

    def __init__(self, min_size_threshold: int):
        self.min_size_threshold = min_size_threshold
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) >= self.min_size_threshold


class MaxLaneletCountFilter(ScenarioFilter):
    """
    Rejects scenarios where the amount of lanelets in the lanelet network is less than the size threshold.
    """

    def __init__(self, max_size_threshold: int):
        self.max_size_threshold = max_size_threshold
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return len(scenario_bundle.preprocessed_scenario.lanelet_network.lanelets) <= self.max_size_threshold
