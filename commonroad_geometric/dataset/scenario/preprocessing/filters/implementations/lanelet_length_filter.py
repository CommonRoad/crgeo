from typing import Optional

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class LaneletLengthFilter(ScenarioFilter):
    """
    Rejects scenarios where lanelets are too short or too long.
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelets = scenario_bundle.preprocessed_scenario.lanelet_network.lanelets
        min_length_scenario = min((lanelet.distance[-1] for lanelet in lanelets))
        max_length_scenario = max((lanelet.distance[-1] for lanelet in lanelets))

        if self.min_length is not None and min_length_scenario < self.min_length:
            return False
        if self.max_length is not None and max_length_scenario > self.max_length:
            return False

        return True
