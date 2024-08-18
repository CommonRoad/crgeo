import torch
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from torch import Tensor, nn, BoolTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Data

from projects.geometric_models.drivable_area.models.modules.deep_set import DeepSetInvariant
from projects.geometric_models.drivable_area.models.modules.mlp import MLP
from commonroad_geometric.common.config import Config


class LaneletNetworkEncoder(nn.Module):

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
                    batch_norm=True,
                    bias=True,
                ),
                output_transform=MLP(
                    channel_list=[64, 128, self.cfg.rnn_hidden_size],
                    dropout=self.cfg.deep_set_mlp_dropout,
                    batch_norm=True,
                    bias=True,
                ),
                aggregation=self.cfg.deep_set_aggregation,
            )
        else:
            raise ValueError(f"Unknown vertex_feature_encoder config value: {self.cfg.vertex_feature_encoder}")

        # self.node_attr_mlp = MLP(
        #     channel_list=[
        #         self.cfg.rnn_hidden_size + self.cfg.lanelet_features,
        #         *[self.cfg.node_feature_size] * self.cfg.node_feature_mlp_layers,
        #     ],
        #     batch_norm=True,
        #     bias=True,
        #     residual_connections=True,
        # )
        self.node_attr_lin = nn.Linear(
            self.cfg.rnn_hidden_size + self.cfg.lanelet_features,
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
            LaneletEdgeType.CONFLICTING
        ]
        self.edge_type_embedding = None
        if self.cfg.edge_type_encoding == "embedding":
            self.node_type_embedding_dim = self.cfg.node_type_embedding_dim
            self.edge_type_embedding = nn.Embedding(
                num_embeddings=len(self.edge_types),
                embedding_dim=self.node_type_embedding_dim,
                # max_norm=1.0,
                # norm_type=2.0,
            )
        elif self.cfg.edge_type_encoding == "one-hot":
            edge_type_onehot = torch.diag(torch.ones(len(self.edge_types), dtype=torch.float32))
            self.register_buffer("edge_type_onehot", edge_type_onehot)
            self.node_type_embedding_dim = len(self.edge_types)
        else:
            raise ValueError(f"Unknown edge_type_encoding value {self.cfg.edge_type_encoding}")

        self.edge_lin = nn.Linear(
            self.node_type_embedding_dim + self.cfg.lanelet_to_lanelet_features,
            self.cfg.edge_feature_size,
            bias=True,
        )

    @property
    def output_feature_size(self) -> int:
        return self.cfg.node_feature_size

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        device = data.relative_vertices.device
        N, E = data.num_nodes, data.edge_index.shape[1]

        # == Node features ==
        vertices = data.relative_vertices.view(N, -1, 4)

        if self.cfg.vertex_feature_encoder in {"LSTM", "GRU"}:
            # encode lanelet vertices with an RNN

            # convert absolute vertex positions to delta positions
            vertices[:, 1:] = vertices[:, 1:] - vertices[:, :-1]

            # version with padded sequences in one Tensor
            vertices_packed = pack_padded_sequence(
                input=vertices,
                lengths=data.vertices_lengths.to(device="cpu").flatten(),
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
            raise ValueError(f"Unknown vertex_feature_encoder config value: {self.cfg.vertex_feature_encoder}")

        x = torch.cat([
            encoder_output,
            data.length,
            data.curvature,
        ], dim=-1)
        x = self.node_attr_lin(x)

        # == Edges ==
        # filter by edge type
        edge_mask = torch.zeros(E, dtype=torch.bool, device=device)
        edge_type_emb = torch.empty(
            E, self.node_type_embedding_dim,
            dtype=torch.float32,
            device=device,
        )
        for index, edge_type in enumerate(self.edge_types):
            edge_type_mask: BoolTensor = (data.type == edge_type).squeeze(-1)
            edge_mask.logical_or_(edge_type_mask)
            edge_type_ind = edge_type_mask.nonzero().squeeze()
            if self.edge_type_embedding is not None:
                edge_type_emb[edge_type_mask] = self.edge_type_embedding(
                    torch.empty_like(edge_type_ind, device=device).fill_(index),
                )
            else:
                edge_type_emb[edge_type_mask] = self.edge_type_onehot[index].expand(edge_type_ind.size(0), -1)

        assert edge_mask.all()
        # edge_index = data.edge_index[:, edge_mask]

        # == Edge features ==
        edge_attr = torch.hstack([
            # edge_type_emb[edge_mask],
            # data.edge_attr[edge_mask],
            edge_type_emb,
            data.source_arclength_rel,
            data.target_arclength_rel,
            data.traffic_flow,
            data.relative_orientation,
            data.relative_position,
            data.target_arclength,
            data.source_arclength,
        ])
        edge_attr = self.edge_lin(edge_attr)

        return x, edge_attr
