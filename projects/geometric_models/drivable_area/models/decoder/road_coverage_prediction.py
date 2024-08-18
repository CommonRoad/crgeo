import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import HEATConv

from commonroad_geometric.common.config import Config
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.learning.geometric.base_geometric import BaseModel
from projects.geometric_models.drivable_area.models.encoder.lanelet_network_encoder import LaneletNetworkEncoder
from projects.geometric_models.drivable_area.models.modules.custom_egat_conv import RoadCoverageEGATConv


class CustomHEATConv(BaseModel):

    def __init__(
        self, *,
        in_channels: int,
        out_channels: int,
        heads: int,
        **kwargs,
    ):
        super().__init__()
        self.heat_conv = HEATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            **kwargs,
        )
        self.lin = nn.Linear(
            heads * out_channels,
            out_channels,
            bias=True,
        )

    def forward(self, *args, **kwargs) -> Tensor:
        out = self.heat_conv(*args, **kwargs)
        out = self.lin(out)
        return out


class LaneletNetworkGNN(BaseModel):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.road_coverage

        self.lanelet_network_encoder = LaneletNetworkEncoder(cfg=cfg.lanelet_network_encoder)

        # == Graph convolutional layers ==
        if self.cfg.gnn_type == "EGATConv":
            self.graph_convs = nn.ModuleList([
                RoadCoverageEGATConv(
                    in_channels=self.cfg.node_feature_size,
                    out_channels=self.cfg.node_feature_size,
                    edge_dim=self.cfg.edge_feature_size,
                )
                for _ in range(self.cfg.gnn_conv_layers)
            ])
        elif self.cfg.gnn_type == "HEATConv":
            self.graph_convs = nn.ModuleList([
                CustomHEATConv(
                    in_channels=self.cfg.node_feature_size,
                    out_channels=self.cfg.node_feature_size,
                    num_node_types=1,
                    num_edge_types=1,
                    edge_dim=self.cfg.edge_feature_size,
                    edge_attr_emb_dim=self.cfg.edge_feature_size,
                    edge_type_emb_dim=1,
                    heads=self.cfg.gnn_attention_heads,
                    concat=True,
                    bias=False,
                )
                for _ in range(self.cfg.gnn_conv_layers)
            ])
        # TODO integrate custom Heterogeneous Graph Transformer layer
        else:
            raise ValueError(f"Unknown gnn_type config value: {self.cfg.gnn_type}")

        # Jumping Knowledge skip connections to stabilize training with more graph convolutional layers
        # and to be able to fuse features from multiple propagation depths
        self.jumping_knowledge = self.cfg.jumping_knowledge
        if self.jumping_knowledge == "disabled":
            # self.gnn_output_lin = nn.Linear(
            #     self.cfg.node_feature_size,
            #     self.cfg.node_output_size,
            #     bias=True,
            # )
            pass
        elif self.jumping_knowledge == "concat":
            self.jumping_knowledge_lin = nn.Linear(
                (1 + len(self.graph_convs)) * self.cfg.node_feature_size,
                self.cfg.node_output_size,
                bias=True,
            )
        else:
            raise ValueError(f"Unknown jumping_knowledge config value: {self.jumping_knowledge}")

    @property
    def output_feature_size(self) -> int:
        return self.cfg.node_output_size

    def forward(self, data: Data) -> Tensor:
        device = self.device
        N, E = data.num_nodes, data.num_edges

        x, edge_index, edge_attr = self.lanelet_network_encoder(data)

        # == Graph convolutions ==
        # update node representations with graph convolutions
        if self.cfg.gnn_type == "HEATConv":
            # HEATConv helpers
            node_type = torch.zeros(N, dtype=torch.long, device=device)
            edge_type = torch.zeros(E, dtype=torch.long, device=device)

        layer_representations = [x]  # for Jumping Knowledge
        for i, graph_conv in enumerate(self.graph_convs):
            if self.cfg.gnn_type == "EGATConv":
                x_new = graph_conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            elif self.cfg.gnn_type == "HEATConv":
                x_new = graph_conv(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_type=node_type,
                    edge_type=edge_type)
            x_new = F.relu(x_new)
            assert_size(x_new, (N, self.cfg.node_feature_size))

            if self.cfg.gnn_skip_connections:
                x = x + x_new
            else:
                x = x_new

            layer_representations.append(x)

        # Jumping knowledge
        if self.jumping_knowledge == "disabled":
            # x = self.gnn_output_lin(x)
            pass
        elif self.jumping_knowledge == "concat":
            x = torch.hstack(layer_representations)
            x = self.jumping_knowledge_lin(x)
            assert_size(x, (N, self.cfg.node_feature_size))
        elif self.jumping_knowledge == "max":
            x = torch.stack(layer_representations)
            x = torch.max(x, dim=0)  # TODO https://proceedings.mlr.press/v80/xu18c/xu18c.pdf page 5
        else:
            assert False

        assert_size(x, (N, self.cfg.node_output_size))
        return x
