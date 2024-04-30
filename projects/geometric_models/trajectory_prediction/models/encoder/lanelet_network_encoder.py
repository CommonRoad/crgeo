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
from projects.geometric_models.drivable_area.models.modules.deep_set import DeepSetInvariant


class LaneletNetworkEncoder(BaseModel):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # == Node features ==
        if self.cfg.vertex_feature_encoder in {"LSTM", "GRU"}:
            rnn_class = nn.LSTM if self.cfg.vertex_feature_encoder == "LSTM" else nn.GRU
            self.lanelet_vertices_encoder = rnn_class(
                input_size=4,
                hidden_size=self.cfg.rnn_hidden_size,
                batch_first=True,  # input and output tensors have shape [batch, sequence, feature]
                bias=True,
            )
        elif self.cfg.vertex_feature_encoder == "DeepSet":
            # "bag of lanelet segments"
            self.lanelet_vertices_encoder = DeepSetInvariant(
                element_transform=MLP(
                    channel_list=[4, 32, 64],
                    dropout=self.cfg.deep_set_mlp_dropout,
                    norm="batch_norm",
                    bias=True,
                ),
                output_transform=MLP(
                    channel_list=[64, 128, self.cfg.rnn_hidden_size],
                    dropout=self.cfg.deep_set_mlp_dropout,
                    norm="batch_norm",
                    bias=True,
                ),
                aggregation=self.cfg.deep_set_aggregation,
            )
        else:
            raise ValueError(f"Unknown vertex_feature_encoder config value: {self.cfg.vertex_feature_encoder}")

        self.node_attr_mlp = MLP(
            channel_list=[
                self.cfg.rnn_hidden_size + self.cfg.lanelet_features,
                *[self.cfg.node_feature_size] * self.cfg.node_feature_mlp_layers,
            ],
            norm="batch_norm",
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

        self.batch_norm_l = BatchNorm(self.cfg.lanelet_features)
        self.batch_norm_l2l = BatchNorm(self.cfg.lanelet_to_lanelet_features)

    @property
    def output_feature_size(self) -> int:
        return self.cfg.node_feature_size

    def forward(self, data: CommonRoadData) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.device
        N, E = data.l.num_nodes, data.l2l.num_edges

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

        else:
            assert False

        x_lanelet_static = torch.cat([
            data.l.length,
            data.l.curvature,
        ], dim=-1)

        x = torch.cat([
            encoder_output,
            self.batch_norm_l(x_lanelet_static)
        ], dim=-1)
        x = self.node_attr_mlp(x)
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
            data.l2l.relative_orientation,
            data.l2l.source_arclength_rel,
            data.l2l.target_arclength_rel,
            data.l2l.relative_position,
            data.l2l.distance
        ], -1)

        # == Edge features ==
        edge_attr = torch.hstack([
            edge_type_emb[edge_mask],
            self.batch_norm_l2l(edge_attr_l2l_static[edge_mask]),
        ])
        edge_attr = self.edge_lin(edge_attr)

        return x, edge_index, edge_attr
