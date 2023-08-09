from __future__ import annotations

from typing import Callable, Optional, Tuple, Type, Union

from commonroad.scenario.lanelet import LaneletNetwork

from commonroad_geometric.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph, CanonicalTransform
from commonroad_geometric.dataset.extraction.road_network.implementations.lanelet_endpoint_graph.graph_conversion import LaneletEndpointGraphConverter


class LaneletEndpointGraph(BaseRoadNetworkGraph):

    @classmethod
    def from_lanelet_network(
            cls,
            lanelet_network: LaneletNetwork,
            conversion_steps: Tuple[Callable] = None,
            decimal_precision: int = 2,
            waypoint_density: Union[float, int] = 50,
            max_iterations: int = 1e6
    ) -> BaseRoadNetworkGraph:
        if conversion_steps is None:
            conversion_steps = (
                LaneletEndpointGraphConverter._insert_conflict_nodes,
                LaneletEndpointGraphConverter._connect_adjacent_nodes,
                LaneletEndpointGraphConverter._add_diagonal_edges
            )

        graph_converter = LaneletEndpointGraphConverter(
            conversion_steps=conversion_steps,
            decimal_precision=decimal_precision,
            waypoint_density=waypoint_density,
            max_iterations=max_iterations
        )

        graph, lanelet_network = graph_converter.create_lanelet_endpoint_graph_from_lanelet_network(
            lanelet_network=lanelet_network
        )
        lanelet_endpoint_graph = cls(graph=graph, lanelet_network=lanelet_network)
        return lanelet_endpoint_graph

    def get_canonical_graph(
        self,
        source: Tuple[int, int],
        include_radius: Optional[float] = 100.0,
        depth: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        distance_offset: float = 0.0,
        transform_mode: Optional[CanonicalTransform] = CanonicalTransform.TranslateRotateRescale
    ) -> Tuple[Type[BaseRoadNetworkGraph], LaneletNetwork]:

        raise NotImplementedError(f"This method has not been implemented for {self.__class__.name}")
