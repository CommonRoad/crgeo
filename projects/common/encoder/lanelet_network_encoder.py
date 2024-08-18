from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn.models import MLP
from torch_geometric.nn.norm import BatchNorm

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.common.modules.deep_set import DeepSetInvariant


class LaneletNetworkEncoder(BaseModel):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.vertex_feature_encoder in ["LSTM", "GRU","DeepSet"], "Vertex feature encode should be either LSTM or GRU or DeepSet"
        self.batch_norm_l = BatchNorm(self.cfg.lanelet_features) if self.cfg.batch_norm_input else nn.Identity()
        self.batch_norm_l2l = BatchNorm(self.cfg.lanelet_to_lanelet_features)  if self.cfg.batch_norm_input else nn.Identity()

        if self.cfg.vertex_feature_encoder_enabled:
            self.rnn_encoder_dim = self.cfg.rnn_hidden_size
            # == Node features ==
            if self.cfg.vertex_feature_encoder in {"LSTM", "GRU"}:
                rnn_class = getattr(nn, self.cfg.vertex_feature_encoder)
                self.lanelet_vertices_encoder = rnn_class(
                    input_size=4,
                    hidden_size=self.cfg.rnn_hidden_size,
                    batch_first=True,  # input and output tensors have shape [batch, sequence, feature]
                    bias=True,
                )
                self.rnn_encoder_dim = self.cfg.rnn_hidden_size

            elif self.cfg.vertex_feature_encoder == "DeepSet":
                # "bag of lanelet segments"
                self.lanelet_vertices_encoder = DeepSetInvariant(
                    element_transform=MLP(
                        channel_list=[4, 32, 64],
                        dropout=self.cfg.deep_set_mlp_dropout,
                        norm="batch_norm" if self.cfg.batch_norm_hidden else None,
                        bias=True,
                    ),
                    output_transform=MLP(
                        channel_list=[64, 128, self.cfg.rnn_hidden_size],
                        dropout=self.cfg.deep_set_mlp_dropout,
                        norm="batch_norm" if self.cfg.batch_norm_hidden else None,
                        bias=True,
                    ),
                    aggregation=self.cfg.deep_set_aggregation,
                )
        else:
            self.lanelet_vertices_encoder = None
            self.rnn_encoder_dim = 0

        # self.node_attr_mlp = MLP(
        #     channel_list=[
        #         self.rnn_encoder_dim + self.cfg.lanelet_features,
        #         *[self.cfg.node_feature_size] * self.cfg.node_feature_mlp_layers,
        #     ],
        #     norm="batch_norm" if self.cfg.batch_norm_hidden else None,
        #     bias=True,
        # )
        self.node_attr_lin = nn.Linear(
            self.rnn_encoder_dim + self.cfg.lanelet_features,
            self.cfg.node_feature_size,
            bias=True,
        )

        # == Edge features ==
        # filter out edges not in the edge_types list
        self.edge_types = [
            LaneletEdgeType.PREDECESSOR,
            LaneletEdgeType.SUCCESSOR,
            LaneletEdgeType.ADJACENT_LEFT,
            LaneletEdgeType.OPPOSITE_LEFT,
            LaneletEdgeType.ADJACENT_RIGHT,
            LaneletEdgeType.OPPOSITE_RIGHT,
            LaneletEdgeType.MERGING,
            LaneletEdgeType.DIVERGING,
        ]
        self.edge_type_embedding = nn.Embedding(
            num_embeddings=len(self.edge_types),
            embedding_dim=self.cfg.node_type_embedding_dim,
            # max_norm=1.0,
            # norm_type=2.0,
        )
        self.edge_lin = nn.Linear(
            self.cfg.node_type_embedding_dim + self.cfg.lanelet_to_lanelet_features,
            self.cfg.edge_feature_size,
            bias=False,
        )

    @property
    def output_feature_size(self) -> int:
        return self.cfg.node_feature_size

    def forward(self, data: CommonRoadData) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.device
        N, E = data.l.num_nodes, data.l2l.edge_index.shape[1]

        x_lanelet_static = torch.cat([
            data.l.length/100,
            data.l.curvature,
            data.l.start_curvature,
            data.l.end_curvature,
            data.l.direction_change
        ], dim=-1)

        if not self.cfg.vertex_feature_encoder_enabled:
            x = x_lanelet_static
        else:
            if N == 0:
                encoder_output = torch.zeros((0, self.cfg.rnn_hidden_size), device=device)
            else:
                # == Node features ==
                vertices = data.l.relative_vertices.view(N, -1, 4)

                if self.cfg.vertex_feature_encoder in {"LSTM", "GRU"}:
                    # encode lanelet vertices with an RNN

                    # convert absolute vertex positions to delta positions
                    vertices[:, 1:] = vertices[:, 1:] - vertices[:, :-1]

                    # version with padded sequences in one Tensor
                    vertices_packed = pack_padded_sequence(
                        input=vertices,
                        lengths=data.l.vertices_lengths.to(device="cpu").flatten(),
                        batch_first=True,
                        enforce_sorted=False,
                    )
                    # version with a Python list of Tensors (i.e. list of sequences)
                    # vertices_packed = pack_sequence(data[attr], enforce_sorted=False)

                    encoder_output, _ = self.lanelet_vertices_encoder(vertices_packed)
                    encoder_output, rnn_output_lengths = pad_packed_sequence(encoder_output, batch_first=True)
                    _node_indices = torch.arange(N, device=device)
                    encoder_output = encoder_output[_node_indices, rnn_output_lengths - 1]

                elif self.cfg.vertex_feature_encoder == "DeepSet":
                    encoder_output = self.lanelet_vertices_encoder(vertices)

            x = torch.cat([
                encoder_output,
                self.batch_norm_l(x_lanelet_static)
            ], dim=-1)

        x = torch.tanh(self.node_attr_lin(x))
        assert_size(x, (N, self.cfg.node_feature_size))

        # == Edges ==
        # filter by edge type
        edge_mask = torch.zeros(E, dtype=torch.bool, device=device)
        edge_type_emb = torch.empty(
            E, self.edge_type_embedding.embedding_dim,
            dtype=self.edge_type_embedding.weight.data.dtype,
            device=device,
        )
        for value, edge_type in enumerate(self.edge_types):
            edge_type_mask: Tensor = data.l2l.type.squeeze(-1) == edge_type
            edge_mask.logical_or_(edge_type_mask)
            edge_type_ind = edge_type_mask.nonzero()
            edge_type_emb[edge_type_ind] = self.edge_type_embedding(
                torch.empty_like(edge_type_ind, device=device).fill_(value),
            )

        edge_index = data.l2l.edge_index[:, edge_mask]

        edge_attr_l2l_static = torch.cat([
            torch.cos(data.l2l.relative_orientation),
            torch.sin(data.l2l.relative_orientation),
            data.l2l.source_arclength_rel,
            data.l2l.target_arclength_rel,
            data.l2l.relative_position / 100,
            data.l2l.distance / 100
        ], -1)

        # == Edge features ==
        edge_attr = torch.hstack([
            edge_type_emb[edge_mask],
            self.batch_norm_l2l(edge_attr_l2l_static[edge_mask]),
        ])
        edge_attr = torch.tanh(self.edge_lin(edge_attr))

        return x, edge_index, edge_attr
