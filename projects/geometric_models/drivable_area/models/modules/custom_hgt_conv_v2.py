import math
from collections.abc import Callable, Sequence
from typing import Optional, Union, Literal, Any, TypeVar, Iterable, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, LayerNorm, ModuleDict, ParameterDict
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax, add_self_loops
from commonroad_geometric.common.type_checking import all_same

from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax, add_self_loops, contains_self_loops

from commonroad_geometric.common.type_checking import all_same
from commonroad_geometric.common.torch_utils.helpers import reset_module

K = TypeVar('K')
V = TypeVar("V")


def common_keys(dicts):
    if not dicts:
        return []  # Return an empty iterator if the list is empty
    
    # Start with the keys of the first dictionary
    common = set(dicts[0])
    
    # Intersect with the keys of each subsequent dictionary
    for d in dicts[1:]:
        common.intersection_update(d)
    
    return list(common)


def zip_dict_items(*dicts: Dict[K, V]) -> Iterable[Tuple[K, V]]:
    if len(dicts) > 0:
        keys = common_keys(dicts)
        for key in keys:
            yield key, *(d[key] for d in dicts)


ActivationFunction = Callable[[Tensor], Tensor]



class CustomHGTConv2(MessagePassing):

    GLOBAL_CONTEXT_NODE: NodeType = "global context"

    def __init__(
        self,
        in_channels_node: dict[NodeType, int],
        in_channels_edge: dict[EdgeType, int],
        attention_channels: int,
        out_channels_node: dict[NodeType, int],
        metadata: Metadata,
        attention_heads: int,
        activation_fn: ActivationFunction = F.gelu,
        enable_relation_prior: bool = True,
        enable_message_norm: bool = True,
        enable_residual_connection: bool = True,
        enable_residual_weights: bool = True,
        add_self_loops: Optional[Sequence[EdgeType]] = None,
        enable_global_context: bool = True,
    ):
        super().__init__(aggr=None, node_dim=0)
        node_types, edge_types = metadata

        if enable_global_context:
            assert self.GLOBAL_CONTEXT_NODE in in_channels_node and self.GLOBAL_CONTEXT_NODE in out_channels_node

            self.global_in_channels = in_channels_node[self.GLOBAL_CONTEXT_NODE]
            self.global_out_channels = out_channels_node[self.GLOBAL_CONTEXT_NODE]

            in_channels_node = in_channels_node.copy()
            out_channels_node = out_channels_node.copy()
            del in_channels_node[self.GLOBAL_CONTEXT_NODE]
            del out_channels_node[self.GLOBAL_CONTEXT_NODE]

            # force all node feature vectors to be of the same size so they can be added up for
            # updating the global context vector
            assert all_same(out_channels_node.values())

        assert set(in_channels_node.keys()) == set(out_channels_node.keys()) == set(node_types)
        assert set(in_channels_edge.keys()) == set(edge_types)
        assert add_self_loops is None or len(add_self_loops) > 0 and all(src == dst for (src, _, dst) in add_self_loops)

        self.in_channels_node: dict[NodeType, int] = in_channels_node
        self.in_channels_edge: dict[EdgeType, int] = in_channels_edge
        self.attention_channels: int = attention_channels
        self.out_channels_node: dict[NodeType, int] = out_channels_node
        assert all(out_channels % attention_heads == 0 for out_channels in out_channels_node.values())

        self.attention_heads = attention_heads
        self.activation_fn = activation_fn
        self.enable_relation_prior = enable_relation_prior
        self.enable_message_norm = enable_message_norm
        self.enable_residual_connection = enable_residual_connection
        self.enable_residual_weights = enable_residual_weights
        self.add_self_loops = add_self_loops
        self.enable_global_context = enable_global_context

        if enable_residual_connection:
            assert all(out_channels_node[node_type] == in_channels_node[node_type] for node_type in node_types), \
                "Output channels have to match input channels for all node types if residual connection is enabled"

        assert attention_channels % attention_heads == 0
        dim_attention = attention_channels // attention_heads

        # node-type-specific parameters
        self.k_node_lin = ModuleDict()
        self.q_node_lin = ModuleDict()
        self.v_node_lin = ModuleDict()
        self.message_layer_norm = ModuleDict()
        self.aggr_lin = ModuleDict()
        self.residual_weight = ParameterDict()  # beta
        self.residual_layer_norm = ModuleDict()
        for node_type, in_channels, out_channels in ((t, self.in_channels_node[t], self.out_channels_node[t]) for t in node_types):
            # matrices for all attention heads combined
            # attention
            self.k_node_lin[node_type] = Linear(in_channels, attention_channels, bias=False)
            self.q_node_lin[node_type] = Linear(in_channels, attention_channels, bias=False)
            # message
            self.v_node_lin[node_type] = Linear(in_channels, out_channels, bias=False)
            if self.enable_message_norm:
                self.message_layer_norm[node_type] = LayerNorm([out_channels])
            # update
            self.aggr_lin[node_type] = Linear(out_channels, out_channels, bias=False)
            if self.enable_residual_weights:
                self.residual_weight[node_type] = Parameter(torch.empty(1, dtype=torch.float32))
            if self.enable_residual_connection:
                self.residual_layer_norm[node_type] = LayerNorm([out_channels])

        # edge-type-specific parameters
        self.k_edge_lin = ModuleDict()
        self.q_edge_lin = ModuleDict()
        self.att_relation_lin = ParameterDict()
        self.v_relation_node_lin = ModuleDict()
        self.v_relation_edge_lin = ModuleDict()
        self.prior = ParameterDict()
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            dim_node = self.out_channels_node[src_type]
            dim_out = self.out_channels_node[dst_type]
            dim_edge = self.in_channels_edge[edge_type]
            relation_type = "__".join(edge_type)
            # attention
            self.k_edge_lin[relation_type] = Linear(dim_edge, attention_channels, bias=False)
            self.q_edge_lin[relation_type] = Linear(dim_edge, attention_channels, bias=False)
            self.att_relation_lin[relation_type] = Parameter(torch.empty((attention_heads, dim_attention, dim_attention), dtype=torch.float32))
            self.prior[relation_type] = Parameter(torch.empty(attention_heads, dtype=torch.float32))
            # message
            self.v_relation_node_lin[relation_type] = Linear(dim_node, dim_out, bias=False)
            self.v_relation_edge_lin[relation_type] = Linear(dim_edge, dim_out, bias=False)

        # parameters for global context
        self.global_init: Optional[Parameter] = None
        self.global_lin: Optional[Linear] = None
        if enable_global_context:
            dim_out = next(iter(out_channels_node.values()))
            self.global_init = Parameter(torch.ones((self.global_in_channels,), dtype=torch.float32))
            self.global_lin = Linear(dim_out, self.global_out_channels, bias=False)
            self.q_node_lin[self.GLOBAL_CONTEXT_NODE] = Linear(self.global_in_channels, attention_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # reset_module(m) calls m.reset_parameters() if it exists, otherwise it resets the modules' children
        reset_module(self.k_node_lin)
        reset_module(self.q_node_lin)
        reset_module(self.v_node_lin)
        reset_module(self.message_layer_norm)
        reset_module(self.aggr_lin)
        ones(self.residual_weight)
        reset_module(self.residual_layer_norm)
        reset_module(self.k_edge_lin)
        reset_module(self.q_edge_lin)
        glorot(self.att_relation_lin)
        reset_module(self.v_relation_node_lin)
        reset_module(self.v_relation_edge_lin)
        ones(self.prior)
        if self.global_lin is not None:
            ones(self.global_init)
            reset_module(self.global_lin)

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: Union[dict[EdgeType, Tensor], dict[EdgeType, SparseTensor]],  # Support both.
        edge_attr_dict: dict[EdgeType, Tensor],
    ) -> dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict: A dictionary holding input node features for each individual node type.
            edge_index_dict: A dictionary holding graph connectivity information for each individual edge type.
            edge_attr_dict: A dictionary holding input edge features for each individual edge type.

        :rtype: - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        # H = #attention heads
        H, D_head = self.attention_heads, self.attention_channels // self.attention_heads
        device: torch.device = next(iter(x_dict.values())).device

        k_node_dict: dict[NodeType, Tensor] = {}
        q_node_dict: dict[NodeType, Tensor] = {}
        v_node_dict: dict[NodeType, Tensor] = {}
        out_aggr_dict: dict[NodeType, list[tuple[Tensor, int, Tensor, Tensor]]] = {}

        node_types = set(x_dict.keys())

        if self.enable_global_context:
            if self.GLOBAL_CONTEXT_NODE in x_dict:
                node_types.remove(self.GLOBAL_CONTEXT_NODE)
            else:
                # add context vector initialized with learnable features
                x_dict[self.GLOBAL_CONTEXT_NODE] = self.global_init

        # Compute the node part of attention vectors: key, query, value
        for node_type, x in ((node_type, x_dict[node_type]) for node_type in node_types):
            k_node_dict[node_type] = self.k_node_lin[node_type](x).view(-1, H, D_head)
            q_node_dict[node_type] = self.q_node_lin[node_type](x).view(-1, H, D_head)
            v_node_dict[node_type] = self.v_node_lin[node_type](x)
            out_aggr_dict[node_type] = []

        if self.enable_global_context:
            q_node_dict[self.GLOBAL_CONTEXT_NODE] = self.q_node_lin[self.GLOBAL_CONTEXT_NODE](x_dict[self.GLOBAL_CONTEXT_NODE])

        # Iterate over edge types / relation types
        for edge_type, edge_index, edge_attr in zip_dict_items(edge_index_dict, edge_attr_dict):
            src_type, _, dst_type = edge_type
            relation_type = "__".join(edge_type)  # identifies the edge type, edge_type[1] is not sufficient
            D_in_edge = self.in_channels_edge[edge_type]
            D_value = self.out_channels_node[dst_type] // self.attention_heads

            # N x H x D_head
            k_node = k_node_dict[src_type]
            q_node = q_node_dict[dst_type]

            # N x H x D_value
            v_node = self.v_relation_node_lin[relation_type](v_node_dict[src_type]).view(-1, H, D_value)

            # Compute the edge part of attention vectors: key, query, value
            if D_in_edge == 0:
                # no edge attributes
                k_edge = torch.zeros((edge_attr.size(0), H, D_head), dtype=edge_attr.dtype, device=device)
                q_edge = torch.zeros((edge_attr.size(0), H, D_head), dtype=edge_attr.dtype, device=device)
                v_edge = torch.zeros((edge_attr.size(0), H, D_value), dtype=edge_attr.dtype, device=device)
            else:
                # E x H x D_head
                k_edge = self.k_edge_lin[relation_type](edge_attr).view(-1, H, D_head)
                q_edge = self.q_edge_lin[relation_type](edge_attr).view(-1, H, D_head)
                # E x H x D_value
                v_edge = self.v_relation_edge_lin[relation_type](edge_attr).view(-1, H, D_value)

            # prior
            if self.enable_relation_prior:
                prior = self.prior[relation_type]
            else:
                prior = torch.ones((self.attention_heads,), dtype=torch.float32, device=device)

            # self-loops
            k_node_self_loop = None
            q_node_self_loop = None
            v_node_self_loop = None
            if self.add_self_loops is not None and edge_type in self.add_self_loops:
                k_node_self_loop = k_node_dict[dst_type]
                q_node_self_loop = q_node_dict[dst_type]
                v_node_self_loop = self.v_relation_node_lin[relation_type](v_node_dict[dst_type]).view(-1, H, D_value)

                # assert not contains_self_loops(edge_index)
                # self-loop edges are appended at the end of the original edge_index
                # 2 x E_sl
                edge_index, _ = add_self_loops(edge_index)

            edge_index_target, size_target, scores, out = self.propagate(
                edge_index,
                k_node=k_node,
                k_node_self_loop=k_node_self_loop,
                k_edge=k_edge,
                att_lin=self.att_relation_lin[relation_type],
                q_node=q_node,
                q_node_self_loop=q_node_self_loop,
                q_edge=q_edge,
                v_node=v_node,
                v_node_self_loop=v_node_self_loop,
                v_edge=v_edge,
                prior=prior,
                size=None,
            )  # N x D_out

            out_aggr_dict[dst_type].append((edge_index_target, size_target, scores, out))

        # TODO global -> nodes edges

        # aggregate node feature vectors
        out_dict: dict[NodeType, Optional[Tensor]] = {}
        for node_type, outs in out_aggr_dict.items():
            edge_index_target = torch.cat([ edge_index_target for edge_index_target, _, _, _ in outs ], dim=0)
            size_targets = [ size_target for _, size_target, _, _ in outs ]
            scores = torch.cat([ scores for _, _, scores, _ in outs ], dim=0)
            v_source = torch.cat([ out for _, _, _, out in outs ], dim=0)
            size_target = size_targets[0]
            assert all(s == size_target for s in size_targets[1:])

            # softmax over attention scores of all incoming edges
            scores = softmax(src=scores, index=edge_index_target, ptr=None, num_nodes=size_target)  # E_sl x H
            v_source = v_source * scores.unsqueeze(-1)  # E_sl x H x D_value

            D_v = v_source.size(-2) * v_source.size(-1)  # D_v = H * D_value
            # "stack" the vectors of all attention heads
            v_source = v_source.view(-1, D_v)  # E_sl x D_v

            # sum aggregation over messages from incoming edges
            out = scatter(src=v_source, index=edge_index_target, dim=self.node_dim, dim_size=size_target, reduce="sum")

            # apply layer normalization on the messages
            if self.enable_message_norm:
                out = self.message_layer_norm[node_type](out)

            # apply non-linearity
            # then map aggregated feature vector to node-type-specific feature distribution of the target node
            out = self.aggr_lin[node_type](self.activation_fn(out))

            # residual connection
            if self.enable_residual_connection:
                assert out.size() == x_dict[node_type].size()
                if self.enable_residual_weights:
                    beta = self.residual_weight[node_type].sigmoid()
                    out = beta * out + (1 - beta) * x_dict[node_type]
                else:
                    out = out + x_dict[node_type]
                out = self.residual_layer_norm[node_type](out)

            out_dict[node_type] = out

        # update global context
        if self.enable_global_context:
            alpha_list, message_list = [], []
            q = self.q_node_lin[self.GLOBAL_CONTEXT_NODE](x_dict[self.GLOBAL_CONTEXT_NODE]).unsqueeze(0)
            sqrt_d = math.sqrt(self.attention_channels)
            for node_type in node_types:
                h = out_dict[node_type]
                k = self.k_node_lin[node_type](h)
                a = (q * k).sum(dim=-1) / sqrt_d
                alpha_list.append(a)

                message_list.append(self.v_node_lin[node_type](h))

            alpha = torch.cat(alpha_list, dim=0)
            alpha = F.softmax(alpha, dim=0)

            message_global = torch.sum(alpha.view(-1, 1) * torch.cat(message_list, dim=0), dim=0)
            out_dict[self.GLOBAL_CONTEXT_NODE] = self.global_lin(self.activation_fn(message_global))

        return out_dict

    # _i = target, _j = source
    # source -> target

    # E = #edges (without self-loops)
    # E_sl = #edges with self-loops (if self-loops are disabled E_sl == E)
    # H = #attention heads
    # D_head = dimension of each attention head

    # noinspection PyMethodOverriding
    def message(
        self,
        k_node_j: Tensor,  # source, E_sl x H x D_head
        k_node_self_loop_j: Optional[Tensor],  # source, E_sl x H x D_head
        k_edge: Tensor,  # E x H x D_head
        att_lin: Tensor,  # H x D_head x D_head
        q_node_i: Tensor,  # target, E_sl x H x D_head
        q_node_self_loop_i: Optional[Tensor],  # target, E_sl x H x D_head
        q_edge: Tensor,  # E x H x D_head
        v_node_j: Tensor,  # source, E_sl x H x D_value
        v_node_self_loop_j: Optional[Tensor],  # source, E_sl x H x D_value
        v_edge: Tensor,  # E x H x D_value
        prior: Tensor,  # H
        edge_index_i: Tensor,  # E_sl
        # ptr: Optional[Tensor],  # non-null when using sparse tensors for the edge index
        size_i: int,
    ) -> tuple[Tensor, int, Tensor, Tensor]:
        E, H, D_head = k_edge.size()
        has_self_loops = E != k_node_j.size(0)

        # E_sl x H x D_head
        if has_self_loops:
            k_j = torch.empty_like(k_node_j)
            k_j[:E] = k_node_j[:E] + k_edge
            k_j[E:] = k_node_self_loop_j[E:]  # self-loops
        else:
            k_j = k_node_j + k_edge

        #  H x E_sl x D_head @ H x D_head x D_head -> H x E_sl x D_head -> E_sl x H x D_head
        k_relation_j = (k_j.transpose(0, 1) @ att_lin).transpose(1, 0)

        # E_sl x H x D_head
        if has_self_loops:
            q_i = torch.empty_like(q_node_i)
            q_i[:E] = q_node_i[:E] + q_edge
            q_i[E:] = q_node_self_loop_i[E:]  # self-loops
        else:
            q_i = q_node_i + q_edge

        # E_sl x H x D_value
        if has_self_loops:
            v_j = torch.empty_like(v_node_j)
            v_j[:E] = v_node_j[:E] + v_edge
            v_j[E:] = v_node_self_loop_j[E:]  # self-loops
        else:
            v_j = v_node_j + v_edge

        # scaled dot-product attention
        scores = (k_relation_j * q_i).sum(dim=-1) * (prior / math.sqrt(D_head))  # E_sl x H

        return edge_index_i, size_i, scores, v_j

    # noinspection PyMethodOverriding
    def aggregate(
        self,
        inputs: Any,
    ) -> Any:
        # skip aggregation
        return inputs
