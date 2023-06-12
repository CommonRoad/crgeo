import math
from contextlib import suppress
from typing import Optional, List

import networkx as nx
import numpy as np
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.dataset.extraction.traffic.feature_computers.base_feature_computer import BaseFeatureComputer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.types import FeatureDict, V2VFeatureParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation


class LaneletDistanceFeatureComputer(BaseFeatureComputer[V2VFeatureParams]):

    def __init__(
        self,
        max_lanelet_distance: float = 60.0,
        max_lanelet_distance_placeholder: float = 60.0 * 1.1
    ):
        super().__init__()
        self._max_lanelet_distance = max_lanelet_distance
        self._max_lanelet_distance_placeholder = max_lanelet_distance_placeholder

    def __call__(
        self,
        params: V2VFeatureParams,
        simulation: BaseSimulation,
    ) -> FeatureDict:
        # lanelet_distance: shortest distance between the vehicles when following lanelet center lines & changing
        #     lanelets, following the direction of the lanelets

        max_lanelet_distance_exceeded = {
            "lanelet_distance": self._max_lanelet_distance_placeholder,
        }

        if np.linalg.norm(params.source_state.position - params.target_state.position) > self._max_lanelet_distance:
            return max_lanelet_distance_exceeded

        source_lanelet = simulation.get_obstacle_lanelet(params.source_obstacle)
        target_lanelet = simulation.get_obstacle_lanelet(params.target_obstacle)
        if source_lanelet is None or target_lanelet is None:
            # TODO this isn't a good solution
            return max_lanelet_distance_exceeded

        source_center_polyline = simulation.get_lanelet_center_polyline(source_lanelet.lanelet_id)
        target_center_polyline = simulation.get_lanelet_center_polyline(target_lanelet.lanelet_id)
        source_arclength = source_center_polyline.get_projected_arclength(position=params.source_state.position)
        target_arclength = target_center_polyline.get_projected_arclength(position=params.target_state.position)

        if source_lanelet.lanelet_id == target_lanelet.lanelet_id:
            distance = abs(source_arclength - target_arclength)
            if distance > self._max_lanelet_distance:
                return max_lanelet_distance_exceeded

            return {
                "lanelet_distance": abs(source_arclength - target_arclength),
            }

        else:
            lanelet_graph: nx.DiGraph = simulation.lanelet_graph
            Node = int

            def weight_fn(s: Node, t: Node, edge_attr: dict) -> Optional[float]:
                if edge_attr["lanelet_edge_type"] == LaneletEdgeType.SUCCESSOR.value:
                    return edge_attr["weight"]

                elif edge_attr["lanelet_edge_type"] == LaneletEdgeType.ADJACENT_LEFT.value or edge_attr["lanelet_edge_type"] == LaneletEdgeType.ADJACENT_RIGHT.value:
                    start_pos = 0.0
                    end_pos: float = lanelet_graph.nodes[t]["length"]
                    if s == source_lanelet.lanelet_id:
                        start_pos = source_arclength
                    if t == target_lanelet.lanelet_id:
                        end_pos = target_arclength
                    orthogonal_distance = np.linalg.norm(lanelet_graph.nodes[s]["start_pos"] - lanelet_graph.nodes[t]["start_pos"])
                    # ignore curvature of lanelet
                    return math.sqrt(orthogonal_distance ** 2 + (end_pos - start_pos) ** 2)

                else:
                    return None

            for s, t in [ (source_lanelet.lanelet_id, target_lanelet.lanelet_id), (target_lanelet.lanelet_id, source_lanelet.lanelet_id) ]:
                with suppress(nx.NetworkXNoPath):
                    distance: float
                    shortest_path: List[Node]
                    distance, shortest_path = nx.single_source_dijkstra(
                        lanelet_graph,
                        source=s,
                        target=t,
                        weight=weight_fn,
                    )
                    break

            else:
                # no path between the vehicles
                return max_lanelet_distance_exceeded

            if shortest_path[0] == source_lanelet.lanelet_id:
                # source to target
                distance += target_arclength - source_arclength
            else:
                # target to source
                assert shortest_path[0] == target_lanelet.lanelet_id
                distance += source_arclength - target_arclength

            if distance > self._max_lanelet_distance:
                return max_lanelet_distance_exceeded

            return {
                "lanelet_distance": distance,
            }
