from collections import defaultdict
from contextlib import suppress
from typing import Set, Union, Dict, Tuple, DefaultDict
import warnings

import networkx as nx
from typing import Optional
import torch
import torch_geometric.data


def softmax(
    src: torch.Tensor,
    index: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0
) -> torch.Tensor:
    r"""Modified version of https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/utils/softmax.html
    that doesn't rely on TorchScript:

    key = softmax(x, batch)
        RuntimeError: nvrtc: error: failed to open libnvrtc-builtins.so.11.1.
        Make sure that libnvrtc-builtins.so.11.1 is installed correctly.
        nvrtc compilation failed: 
    """
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_scatter import scatter
    N = maybe_num_nodes(index, num_nodes)
    src_max = scatter(src, index, dim, dim_size=N, reduce='max')
    src_max = src_max.index_select(dim, index)
    out = (src - src_max).exp()
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
    out_sum = out_sum.index_select(dim, index)

    return out / (out_sum + 1e-16)


def get_batch_masks(batch: torch.Tensor, max_num_nodes: Optional[int] = None) -> torch.Tensor:
    # TODO: Documentation
    batch_size = int(batch.max().item() + 1)
    graph_sizes = get_batch_sizes(batch)
    max_num_nodes = max_num_nodes if max_num_nodes is not None else int(graph_sizes.max().item())
    graph_arange = torch.arange(max_num_nodes, device=batch.device)
    batch_masks = graph_arange[None, :].repeat(
        batch_size, 1
    ) < graph_sizes[:, None]
    return batch_masks


def get_batch_sizes(batch: torch.Tensor) -> torch.Tensor:
    # TODO: Documentation
    return torch.diff(
        torch.where(torch.cat(
            [torch.diff(batch, prepend=torch.as_tensor([0], device=batch.device)),
            torch.as_tensor([1], device=batch.device)]
        ))[0], prepend=torch.as_tensor([0], device=batch.device)
    )


def get_batch_start_indices(val: torch.Tensor, is_batch_sizes: bool = False) -> torch.Tensor:
    batch_sizes = val if is_batch_sizes else get_batch_sizes(val)
    return torch.cat([torch.tensor([0], device=val.device), batch_sizes]).cumsum(0)[:-1]


def get_batch_internal_indices(batch: torch.Tensor) -> torch.Tensor:
    # TODO: Documentation
    a = torch.arange(len(batch), device=batch.device)
    d = torch.diff(batch, prepend=batch.new_zeros([1]))
    return a - torch.cummax(a*d, 0)[0]


def nx_graph_to_pyg_data(
    graph: Union[nx.Graph, nx.DiGraph],
    convert_attributes_to_tensor: bool = False,
    skip_node_attr: Optional[Set[str]] = None,
    skip_edge_attr: Optional[Set[str]] = None,
    rename_edge_attr: Optional[Dict[str, str]] = None,
    rename_node_attr: Optional[Dict[str, str]] = None,
    node_attr_prefix: str = "",
    edge_attr_prefix: str = "",
) -> Tuple[torch_geometric.data.Data, Dict[int, int]]:
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    This is a custom version of torch_geometric.utils.from_networkx.
    The following modifications have been made:
    - Explicit lanelet id to lanelet index mapping
    - Do not auto-convert attributes to torch.Tensor
    - Allow for skipping and renaming node and edge attributes
    - Add configurable prefix to node and edge attribute names
    - Remove attribute grouping
    """
    if skip_node_attr is None:
        skip_node_attr = set()
    if skip_edge_attr is None:
        skip_edge_attr = set()
    if rename_node_attr is None:
        rename_node_attr = {}
    if rename_edge_attr is None:
        rename_edge_attr = {}

    # relabel nodes with integer labels from 0 to len(graph.nodes) - 1
    # TODO: why are we sorting the nodes?
    sorted_node_ids = sorted(graph.nodes, key=lambda node_id: int(node_id.split('-')[0]) if isinstance(node_id, str) else node_id)
    lanelet_id_to_index_mapping = {
        node_id: idx
        for idx, node_id in enumerate(sorted_node_ids)
    }
    graph = nx.relabel_nodes(graph, mapping=lanelet_id_to_index_mapping, copy=True)

    graph = graph.to_directed() if not nx.is_directed(graph) else graph

    if graph.number_of_nodes() > 0:
        node_attrs = set(str(k) for k in next(iter(graph.nodes(data=True)))[-1].keys())
    else:
        node_attrs = set()

    if graph.number_of_edges() > 0:
        edge_attrs = set(str(k) for k in next(iter(graph.edges(data=True)))[-1].keys())
    else:
        edge_attrs = set()

    # check that node and edge attribute names are unique
    data_node_attrs = set(node_attr_prefix + rename_node_attr.get(attr, attr) for attr in node_attrs)
    data_edge_attrs = set(edge_attr_prefix + rename_edge_attr.get(attr, attr) for attr in edge_attrs)
    duplicate_attrs = data_node_attrs.intersection(data_edge_attrs)
    if duplicate_attrs:
        raise ValueError(f"Some node and edge attribute names are not unique: {duplicate_attrs}")
    all_data_attrs = data_node_attrs.union(data_edge_attrs)
    if "edge_index" in all_data_attrs:
        raise ValueError("edge_index node or edge attribute already exists, would be silently overwritten")

    pyg_data: DefaultDict[str, Union[list, torch.Tensor]] = defaultdict(list)

    # add node attributes
    nodes = sorted(graph.nodes(data=True), key=lambda node: node[0])  # nodes sorted by node id
    for i, (_, feat_dict) in enumerate(nodes):
        if set(feat_dict.keys()) != node_attrs:
            raise ValueError("Not all nodes contain the same attributes")
        for key, value in feat_dict.items():
            if key in skip_node_attr:
                continue
            key = str(key)
            attr = node_attr_prefix + rename_node_attr.get(key, key)
            pyg_data[attr].append(value)

    # add edge attributes
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(graph.edges(data=True, keys=False))
    else:
        edges = list(graph.edges(data=True))
    for i, (_, _, feat_dict) in enumerate(edges):
        if set(feat_dict.keys()) != edge_attrs:
            raise ValueError("Not all edges contain the same attributes")
        for key, value in feat_dict.items():
            if key in skip_edge_attr:
                continue
            key = str(key)
            attr = edge_attr_prefix + rename_edge_attr.get(key, key)
            pyg_data[attr].append(value)

    if convert_attributes_to_tensor:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for key, value in pyg_data.items():
                with suppress(ValueError):
                    pyg_data[key] = torch.tensor(value)

    edge_index = torch.tensor(
        data=[
            (u, v)
            for u, v, _ in edges
        ],
        dtype=torch.long
    ).t().contiguous()
    pyg_data["edge_index"] = edge_index

    pyg_data: torch_geometric.data.Data = torch_geometric.data.Data.from_dict(pyg_data)

    if pyg_data.x is None and pyg_data.pos is None:
        pyg_data.num_nodes = graph.number_of_nodes()

    return pyg_data, lanelet_id_to_index_mapping


def transitive_edges(*edge_indexes: torch.LongTensor) -> torch.LongTensor:
    assert len(edge_indexes) > 0
    if len(edge_indexes) == 1:
        return edge_indexes[0]

    edge_indices = edge_indexes[0][1]
    for next_edge_index in edge_indexes[1:]:
        indices = (edge_indices.unsqueeze(-1) == next_edge_index[0]).nonzero()[:, 1]
        assert torch.all(edge_indices == next_edge_index[0, indices])
        edge_indices = next_edge_index[1, indices]

    transitive_edge_index: torch.LongTensor = torch.stack([
        edge_indexes[0][0],
        edge_indices,
    ], dim=0)
    return transitive_edge_index
