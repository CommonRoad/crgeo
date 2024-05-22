from commonroad_geometric.common.io_extensions.lanelet_network import get_intersection_successors
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


class IntersectionFilter(ScenarioFilter):
    """
    Combination of MinIntersectionFilter, MaxIntersectionFilter, CompleteIntersectionFilter.
    """
    def __init__(
        self,
        max_intersection: int,
        min_intersections: int = 1,
        keep_only_complete_intersections: bool = True
    ) -> None:
        min_filter = MinIntersectionFilter(min_intersections)
        max_filter = MaxIntersectionFilter(max_intersection)
        self.combined_filter: ScenarioFilter = min_filter & max_filter
        if keep_only_complete_intersections:
            self.combined_filter &= CompleteIntersectionFilter()
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        return self.combined_filter._filter(scenario_bundle)


class MinIntersectionFilter(ScenarioFilter):
    """
    Rejects scenarios with too few intersections.
    """
    def __init__(self, min_intersections: int = 1) -> None:
        self.min_intersections = min_intersections
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        num_intersections = len(scenario_bundle.preprocessed_scenario.lanelet_network.intersections)
        return num_intersections >= self.min_intersections


class MaxIntersectionFilter(ScenarioFilter):
    """
    Rejects scenarios with too many intersections.
    """
    def __init__(self, max_intersections: int) -> None:
        self.max_intersections = max_intersections
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        num_intersections = len(scenario_bundle.preprocessed_scenario.lanelet_network.intersections)
        return num_intersections <= self.max_intersections


class CompleteIntersectionFilter(ScenarioFilter):
    """
    Rejects scenarios with incomplete intersections.
    An intersection is considered incomplete when any intersection successor does not have a predecessor or successor.
    """

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelet_network = scenario_bundle.preprocessed_scenario.lanelet_network

        for intersection in lanelet_network.intersections:
            successors = get_intersection_successors(intersection)
            for lanelet_id in successors:
                lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
                if not lanelet.successor or not lanelet.predecessor:
                    return False

        return True
