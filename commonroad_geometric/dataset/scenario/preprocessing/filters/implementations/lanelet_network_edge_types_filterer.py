from typing import Set

from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters import ScenarioFilter


class LaneletNetworkEdgeTypesFilterer(ScenarioFilter):
    def __init__(
        self,
        required_edges_types: Set[LaneletEdgeType]
    ):
        self.required_edges_types = required_edges_types
        super().__init__()

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        lanelet_graph = LaneletGraph.from_scenario(scenario_bundle.preprocessed_scenario)
        included_edge_types = set([LaneletEdgeType(e[2]['lanelet_edge_type']) for e in lanelet_graph.edges(data=True)])
        return len(self.required_edges_types - included_edge_types) == 0
