from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Type

import networkx as nx
if TYPE_CHECKING:
    import torch_geometric.data

from crgeo.dataset.extraction.base_extractor import BaseExtractionParams, BaseExtractor, BaseExtractorOptions
from crgeo.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph, CanonicalTransform

if TYPE_CHECKING:
    from crgeo.simulation.base_simulation import BaseSimulation


@dataclass
class RoadNetworkExtractorOptions(BaseExtractorOptions):
    graph_cls: Type[BaseRoadNetworkGraph]
    min_size: int = 3
    max_size: Optional[int] = None
    depth: Optional[int] = None
    include_radius: Optional[float] = None
    exclude_leaf_nodes: Optional[bool] = True
    transform_mode: Optional[CanonicalTransform] = CanonicalTransform.TranslateRotateRescale
    plot: bool = False
    plot_dir: Optional[str] = None


@dataclass
class RoadNetworkExtractionParams(BaseExtractionParams):
    ...


class RoadNetworkExtractor(BaseExtractor[RoadNetworkExtractionParams, Optional[torch_geometric.data.Data]]):
    def __init__(
        self,
        simulation: BaseSimulation,
        options: RoadNetworkExtractorOptions,
    ) -> None:
        super().__init__(simulation, options=options)
        self._options = options

        self._graph = self._options.graph_cls.from_scenario(self.scenario)
        self._length = len(self._graph.edge_index_dict) or len(self._graph.edge_index_dict_full)
        self._edge_index_dict = self._graph.edge_index_dict or self._graph.edge_index_dict_full
        self._edge_mapping = self._graph._edge_mapping or self._graph._edge_mapping_full
        self._entry_nodes = nx.get_node_attributes(self._graph, 'is_entry')

    def extract(self, params: RoadNetworkExtractionParams) -> Optional[torch_geometric.data.Data]:
        edge_idx = params.index

        source_edge = self._edge_index_dict[edge_idx]
        source_lanelet_id = self._edge_mapping[source_edge]

        # Hacky solution to tackle structural difference issue of LaneletGraphs and IntersectionGraphs
        if isinstance(source_lanelet_id, set) and len(source_lanelet_id) == 0:
            return None
        source_lanelet_id = list(source_lanelet_id)[0] if isinstance(source_lanelet_id, set) else source_lanelet_id # to handle the issue with LaneletGraph

        if self._options.exclude_leaf_nodes and self._entry_nodes.get(source_edge[1], False):
            return None

        try:
            canonical_graph, intermediate_lanelet_network = self._graph.get_canonical_graph(
                include_radius=self._options.include_radius,
                source=source_edge,
                depth=self._options.depth,
                min_size=self._options.min_size,
                max_size=self._options.max_size,
                transform_mode=self._options.transform_mode
            )
        except Exception as e:
            print(f"Exception occured during canonical conversion: {str(e)}")
            return None

        canonical_source_edge = canonical_graph._edge_mapping_inv_full[source_lanelet_id]
        if self._options.exclude_leaf_nodes and \
            canonical_graph.out_degree(canonical_source_edge[-1]) == 0:
            return None

        if canonical_graph is not None:
            data = canonical_graph.get_torch_data()
            if self._options.plot:
                plot_dir = self._options.plot_dir
                orig_dir = plot_dir + "/orig" if plot_dir is not None else plot_dir
                canonical_dir = plot_dir + "/canonical" if plot_dir is not None else plot_dir
                # self._graph.plot(show=False, output_dir=orig_dir)
                canonical_graph.plot(show=False, output_dir=canonical_dir) # plot_kwargs_graph=dict(show_waypoints=False),
        else:
            data = None
        return data

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> BaseExtractor:
        self._iter_counter = 0
        return self

    def __next__(self) -> Optional[torch_geometric.data.Data]:
        """Yields next data instance"""
        if self._iter_counter > len(self):
            raise StopIteration()
        data = self.extract(self._iter_counter)
        self._iter_counter += 1
        return data
