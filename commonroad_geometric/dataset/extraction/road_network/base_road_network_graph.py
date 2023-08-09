from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np
import torch
from commonroad.geometry.transform import translate_rotate
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.torch_utils.helpers import to_float32, to_padded_sequence
from commonroad_geometric.common.torch_utils.pygeo import nx_graph_to_pyg_data
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, TrafficFlowEdgeConnections
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph
from commonroad_geometric.plotting.plot_scenario import plot_scenario


class CanonicalRepresentationError(ValueError):
    pass


class CanonicalTransform(Enum):
    Translate = 'Translate'
    TranslateRescale = 'TranslateRescale' # Newly added transformation option. TranslateRotateRescale provides some weird results?? TODO: try to solve that issue
    TranslateRotate = 'TranslateRotate'
    TranslateRotateRescale = 'TranslateRotateRescale'


def canonical_graph_transform(
    mode: CanonicalTransform,
    pos: np.ndarray,
    origin: np.ndarray,
    end: Optional[np.ndarray] = None,
    distance_offset: float = 0.0,
    scaling_coefficients: Tuple[float, float] = (100.0, 100.0)
) -> Tuple[np.ndarray, float]:
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    translation = -origin
    if mode in {CanonicalTransform.TranslateRotate, CanonicalTransform.TranslateRotateRescale}:
        assert end is not None
        rotation = -np.arctan2(
            -(end[0] - origin[0]),
            (end[1] - origin[1]),
        )
    else:
        rotation = 0.0
    pos = translate_rotate(pos, translation, rotation)

    pos -= np.array([0.0, distance_offset])

    if mode in {CanonicalTransform.TranslateRescale, CanonicalTransform.TranslateRotateRescale}:
        # scaling_factor = 1/np.linalg.norm(end - origin)
        pos /= np.array([scaling_coefficients[0], scaling_coefficients[1]])

    return pos, rotation


class BaseRoadNetworkGraph(ABC, nx.DiGraph, AutoReprMixin):
    # TODO: General cleanup
    # TODO: Documentation
    # TODO: Proper type hints with generics

    def __init__(
        self,
        graph: nx.DiGraph,
        lanelet_network: Optional[LaneletNetwork] = None
    ) -> None:
        super(BaseRoadNetworkGraph, self).__init__()
        self._lanelet_network = lanelet_network
        self._scenario = None
        self.__dict__.update(graph.__dict__)
        self._proximity_matrix = {x[0]: x[1] for x in nx.all_pairs_shortest_path_length(graph.to_undirected())}
        self._node_attr = self.nodes(data=True)._nodes
        self._edge_attr = {(n, m): edge_attr for n, m, edge_attr in self.edges(data=True)}
        self._edge_mapping = {k: v for k, v in nx.get_edge_attributes(graph, 'lanelet_id').items() if v > 0}
        self._edge_mapping_inv = {v: k for k, v in nx.get_edge_attributes(graph, 'lanelet_id').items() if v > 0}
        self._edge_mapping_full = {k: set(v) for k, v in nx.get_edge_attributes(graph, 'lanelets').items()}
        self._edge_mapping_inv_full: Dict[int, Tuple[int, int]] = {}
        for k, v in self._edge_mapping_full.items():
            for lanelet_id in v:
                self._edge_mapping_inv_full[lanelet_id] = k
        self._node_lanelets_mapping = {k: v for k, v in nx.get_node_attributes(graph, 'lanelets').items()}
        self._included_lanelet_ids = set(sum(self._node_lanelets_mapping.values(), ()))
        self._nx_graph = graph

    @property
    def lanelet_network(self) -> Optional[LaneletNetwork]:
        return self._lanelet_network

    @lanelet_network.setter
    def lanelet_network(self, value: LaneletNetwork) -> None:
        self._lanelet_network = value

    @property
    def scenario(self) -> Optional[Scenario]:
        return self._scenario

    @scenario.setter
    def scenario(self, value: Scenario) -> None:
        self._scenario = value

    def get_traffic_flow_graph(self) -> nx.DiGraph:
        return nx.DiGraph(((u, v, e) for u, v, e in self.edges(data=True) if e['lanelet_edge_type'] in TrafficFlowEdgeConnections))

    def get_canonical_graph_from_lanelet_id(
            self,
            lanelet_id: int,
            *args, **kwargs
    ) -> Type[BaseRoadNetworkGraph]:
        if self.lanelet_network is None:
            raise ValueError("LaneletNetwork not set for IsometricGraph!")

        lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
        if not lanelet.successor:
            raise ValueError(f"Lanelet {lanelet_id} has no successor!")
        source = self._edge_mapping_inv[lanelet_id]
        return self.get_canonical_graph(
            source, *args, **kwargs
        )[0]

    def _get_canonical_nx_graph(
        self,
        source: Tuple[int, int],
        depth: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        radius: Optional[float] = None,
        source_pos: Optional[np.ndarray] = None,
        only_forward_edges: bool = True,
    ) -> nx.DiGraph:
        nodes = [source[0], source[1]]
        edges = [source]

        all_edges = [e for e in nx.edge_bfs(
            self, source=source[1], orientation='ignore'
        )]
        pos = nx.get_node_attributes(self, 'node_position')
        if radius is not None:
            source_lanelet_length = np.linalg.norm(np.array(pos[source[0]]) - np.array(pos[source[1]]))
            if source_lanelet_length > radius:
                raise CanonicalRepresentationError(
                    "Source lanelet longer than radius"
                )

        if source_pos is None:
            source_pos = pos[source[0]]

        if source[0] == source[1]:
            raise CanonicalRepresentationError(
                "Source collapsed"
            )

        node_distances = dict(nx.shortest_path_length(self.to_undirected(), weight='weight', source=source[0]))

        if radius is None:
            relevant_edges = all_edges
        else:
            relevant_edges = []
            for e in all_edges:
                start_pos_dist = node_distances[e[0]]
                end_pos_dist = node_distances[e[1]]
                if start_pos_dist < radius or end_pos_dist < radius:
                    relevant_edges.append(e)

        done = False

        all_nodes = list(sum([e[:2] for e in relevant_edges], ()))
        for n in all_nodes:
            if depth is not None and \
                    self._proximity_matrix[source[0]][n] > depth and \
                    self._proximity_matrix[source[1]][n] > depth:
                continue
            if n in nodes:
                continue

            if not done:
                nodes.append(n)
                if max_size is not None and len(nodes) >= max_size:
                    done = True

        for e in relevant_edges:
            if e[0] in nodes and e[1] in nodes:
                if e not in edges:
                    # if (edge[1], edge[0]) in edges:
                    #     raise CanonicalRepresentationError(
                    #         f"Adding reverse edge {edge}"
                    #     )
                    edges.append((e[0], e[1]))

        if min_size is not None and len(nodes) < min_size:
            raise CanonicalRepresentationError(
                "Insufficient number of nodes available for canonical representation"
            )

        canonical_graph = nx.DiGraph()
        edge_nodes = set(sum(edges, ()))
        node_attr = {}
        nodes_insert = []
        for n in nodes:
            if n in edge_nodes:
                nodes_insert.append(n)
                node_attr[n] = {}
                node_attr[n].update(self._node_attr[n])
                node_attr[n]['source'] = int(n in source)

        canonical_graph.add_nodes_from(nodes_insert)
        canonical_graph.add_edges_from(edges)

        edge_attr = {}
        for i, e in enumerate(edges):
            edge_attr[e] = self._edge_attr[e]
            # if i > 0:

            #     assert set(edge_attr[e].keys()) == set(self._edge_attr[edge_mapping[edges[i - 1]]].keys())
        nx.set_node_attributes(canonical_graph, node_attr)
        nx.set_edge_attributes(canonical_graph, edge_attr)

        if only_forward_edges:
            copy_graph = canonical_graph.copy()
            copy_graph.remove_edge(source[0], source[1])
            if (source[1], source[0]) in copy_graph.edges:
                copy_graph.remove_edge(source[1], source[0])
            forward_edges = [source] + [e[:2] for e in nx.edge_bfs(
                copy_graph, source=source[1], orientation='ignore'
            )]
            forward_edge_nodes = set(sum(forward_edges, ()))
            for e in edges:
                if e not in forward_edges:
                    canonical_graph.remove_edge(*e)
            for n in edge_nodes:
                if n not in forward_edge_nodes:
                    canonical_graph.remove_node(n)

        return canonical_graph

    @abstractmethod
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
        ...


    def get_torch_data(
        self,
        source: Optional[Union[int, Tuple[int, int]]] = None,
        max_size: Optional[int] = None,
        **kwargs
    ) -> Data:
        if source is None:
            graph = self
        else:
            if isinstance(source, int):
                graph = self.get_canonical_graph_from_lanelet_id(
                    lanelet_id=source,
                    max_size=max_size,
                    **kwargs
                )
            else:
                graph, _ = self.get_canonical_graph(
                    source=source,
                    max_size=max_size,
                    **kwargs
                )

        # TODO Improve, which attributes should be removed here?
        for _, _, d in graph.edges(data=True):
            d.pop('lanelets', None)
            lanelet_edge_type = d.pop('lanelet_edge_type', LaneletEdgeType.SUCCESSOR.value)
            d['lanelet_edge_type'] = lanelet_edge_type
            d['traffic_flow'] = lanelet_edge_type in TrafficFlowEdgeConnections
        for _, d in graph.nodes(data=True):
            d.pop('', None)

        graph_pyg, _ = nx_graph_to_pyg_data(
            graph=graph._nx_graph,
            convert_attributes_to_tensor=False,
            skip_node_attr={"node_position"},
            rename_node_attr={
                "lanelet_id": "id",
            },
            rename_edge_attr={
                "lanelet_edge_type": "type"
            },
            edge_attr_prefix="edge_attr_"
        )
        if 'relative_left_vertices' in graph_pyg.keys:
            graph_pyg.vertices_lengths = torch.tensor([a.shape[0] for a in graph_pyg.relative_left_vertices], dtype=torch.int32).unsqueeze(-1)
            graph_pyg.relative_vertices = to_padded_sequence([
                np.concatenate([rlv, rrv], axis=-1)
                for rlv, rrv in zip(graph_pyg.relative_left_vertices, graph_pyg.relative_right_vertices)
            ])
            del graph_pyg.relative_left_vertices
            del graph_pyg.relative_right_vertices
        if 'left_vertices' in graph_pyg.keys:
            graph_pyg.left_vertices = to_padded_sequence(graph_pyg.left_vertices)
            graph_pyg.center_vertices = to_padded_sequence(graph_pyg.center_vertices)
            graph_pyg.right_vertices = to_padded_sequence(graph_pyg.right_vertices)

        for attr in graph_pyg.keys:
            if isinstance(graph_pyg[attr], list) and isinstance(graph_pyg[attr][0], np.ndarray):
                graph_pyg[attr] = torch.vstack([torch.from_numpy(row) for row in graph_pyg[attr]])
            else:
                graph_pyg[attr] = torch.tensor(np.array(graph_pyg[attr]))
            if graph_pyg[attr].ndim == 1:
                graph_pyg[attr] = graph_pyg[attr].unsqueeze(-1)

        graph_pyg = to_float32(graph_pyg)
        return graph_pyg

    @classmethod
    def from_data(cls,
        data: Data,
        node_attrs: Optional[Iterable[str]] = ['node_position', 'source'],
        edge_attrs: Optional[Iterable[str]] = ['weight']
    ) -> Type[BaseRoadNetworkGraph]:
        assert not hasattr(data, 'batch') or data.batch.max() == 0
        graph = to_networkx(data=data, node_attrs=node_attrs, edge_attrs=edge_attrs)
        return cls(graph=graph)

    def save(self, file_path: str) -> None:
        from commonroad_geometric.common.utils.filesystem import save_pickle
        save_pickle(self, file_path=file_path)

    @classmethod
    def load(cls, file_path: str) -> Type[BaseRoadNetworkGraph]:
        from commonroad_geometric.common.utils.filesystem import load_pickle
        return load_pickle(file_path=file_path)

    @classmethod
    def from_scenario_file(
        cls,
        file_path: str,
        **parameters
    ) -> BaseRoadNetworkGraph:
        from commonroad.common.file_reader import CommonRoadFileReader
        scenario, _ = CommonRoadFileReader(file_path).open()
        graph = cls.from_scenario(scenario, **parameters)
        return graph

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        **parameters
    ) -> BaseRoadNetworkGraph:
        graph = cls.from_lanelet_network(scenario.lanelet_network, **parameters)
        graph.scenario = scenario
        return graph

    @classmethod
    @abstractmethod
    def from_lanelet_network(
        cls,
        lanelet_network: LaneletNetwork,
        **parameters
    ) -> BaseRoadNetworkGraph:
        ...

    def copy(self, as_view=False):
        graph_copy = self._nx_graph.copy()
        return self.__class__(graph=graph_copy,
                              lanelet_network=self.lanelet_network)

    def subgraph(self, nbunch):
        subgraph = self._nx_graph.subgraph(nbunch)
        # TODO Reduce lanelet network according to nbunch as well
        brn_graph = self.__class__(graph=subgraph,
                                   lanelet_network=self.lanelet_network)
        return brn_graph

    def plot(
        self,
        title: bool = True,
        show: bool = True,
        separate_figures: bool = False,
        output_dir: str = None,
        output_filetype: str = 'pdf',
        plot_kwargs_scenario: Dict[str, Any] = None,
        plot_kwargs_graph: Dict[str, Any] = None,
        figsize: Tuple[int, int] = (16, 12)
    ) -> Figure:
        # TODO: Add documentation
        
        import os
        import matplotlib.pyplot as plt

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        plot_kwargs_scenario = plot_kwargs_scenario or {}
        plot_kwargs_graph = plot_kwargs_graph or {}

        if not separate_figures:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

            plot_scenario(
                self.lanelet_network,
                ax=axes[0],
                title=False,
                **plot_kwargs_scenario
            )
            if title:
                axes[0].set_title("Lanelet network")
            plot_road_network_graph(
                self,
                ax=axes[1],
                **plot_kwargs_graph,
                node_size=figsize[0]*5
            )
            if title:
                axes[1].set_title(f"{self.__class__.__name__}")
            fig.tight_layout()
            plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
            plt.subplots_adjust(wspace=0.1)
            if show:
                plt.show()
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{self.scenario.scenario_id}-{self.__class__.__name__}-subplots.{output_filetype}")
                fig.savefig(output_path)
            return fig

        else:
            scenario_fig = plot_scenario(
                self.lanelet_network,
                title=False,
                **plot_kwargs_scenario
            )
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{self.scenario.scenario_id}.{output_filetype}")
                scenario_fig.savefig(output_path)
            
            graph_fig = plot_road_network_graph(
                self,
                **plot_kwargs_graph,
                node_size=figsize[0]*5
            )
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{self.scenario.scenario_id}-{self.__class__.__name__}.{output_filetype}")
                graph_fig.savefig(output_path)

            if show:
                plt.show()

            return graph_fig

    @property
    def node_index_dict(self) -> Dict[int, int]:
        return {i: n for i, n in enumerate(self.nodes())}

    @property
    def edge_index_dict(self) -> Dict[int, int]:
        return {i: e for i, e in enumerate(self._edge_mapping.keys())}

    @property
    def edge_index_dict_full(self) -> Dict[int, int]:
        return {i: e for i, e in enumerate(self._edge_mapping_full.keys())}

    @property
    def node_index_dict_inv(self) -> Dict[int, int]:
        return {n: i for i, n in enumerate(self.nodes())}

    @property
    def edge_index_dict_inv(self) -> Dict[int, int]:
        return {e: i for i, e in enumerate(self._edge_mapping.keys())}

    @property
    def edge_index_dict_inv_full(self) -> Dict[int, int]:
        return {e: i for i, e in enumerate(self._edge_mapping_full.keys())}
