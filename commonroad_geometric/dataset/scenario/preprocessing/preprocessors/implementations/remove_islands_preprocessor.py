from __future__ import annotations

import networkx as nx

from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_graph.lanelet_graph import LaneletGraph
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor


class RemoveIslandsPreprocessor(ScenarioPreprocessor):
    """
    Removes disconnected parts of the road network by keeping the 
    largest cluster (by number of lanelets).
    """

    def __init__(
        self,
    ) -> None:
        super(RemoveIslandsPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        scenario = scenario_bundle.preprocessed_scenario
        lanelet_graph = LaneletGraph.from_lanelet_network(scenario.lanelet_network)

        connected_components = list(nx.connected_components(nx.to_undirected(lanelet_graph)))
        to_remove = []
        for component in connected_components[1:]:
            # removing all subgraphs except for the largest one
            for lanelet_id in component:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                scenario.lanelet_network.remove_lanelet(lanelet_id)
                to_remove.append(lanelet)
        scenario.remove_hanging_lanelet_members(to_remove)
        scenario.lanelet_network.cleanup_lanelet_references()

        for intersection in scenario.lanelet_network.intersections:
            keep = True
            for incoming_element in intersection.incomings:
                if len(incoming_element.incoming_lanelets) == 0:
                    keep = False
                    break
            if not keep:
                scenario.lanelet_network.remove_intersection(intersection.intersection_id)

        return [scenario_bundle]
