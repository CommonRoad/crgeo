from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class HighwayFilter(ScenarioFilter):
    """
    Rejects scenario with highway-like multi-lanes.
    """

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        # enforce minimum requirements for a scenario

        has_highway: bool = False

        for lanelet in scenario_bundle.preprocessed_scenario.lanelet_network.lanelets:
            if (lanelet.adj_left and lanelet.adj_left_same_direction) or \
                (lanelet.adj_right and lanelet.adj_right_same_direction):
                has_highway = True
                break

        return not has_highway
