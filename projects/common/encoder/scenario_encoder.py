from typing import Dict, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import EdgeType

from commonroad_geometric.common.config import Config
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from projects.common.encoder.lanelet_network_encoder import LaneletNetworkEncoder
from projects.common.modules.custom_hgt_conv import CustomHGTConv
from projects.common.modules.time2vec import Time2Vec

NODE_TYPES = [
    "vehicle",
    "lanelet",
]
EDGE_TYPES = [
    ("vehicle", "to", "vehicle"),
    ("lanelet", "to", "lanelet"),
    ("vehicle", "to", "lanelet"),
    ("lanelet", "to", "vehicle"),
]
EDGE_TYPES_TEMPORAL = EDGE_TYPES + [
    ("vehicle", "temporal", "vehicle"),
]


class ScenarioEncoderModel(nn.Module):
    """Produces a vehicle representation"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.traffic
        self.cfg_graph_features = cfg.graph_features

        # == Lanelet nodes and lanelet-to-lanelet edges ==
        self.lanelet_encoder = LaneletNetworkEncoder(cfg=cfg.lanelet_network_encoder)
        assert self.lanelet_encoder.output_feature_size == self.cfg.lanelet_node_feature_size
        # freeze_weights(self.lanelet_encoder)  # TODO ensure that weights are still frozen after .load_state_dict has been called

        self.vehicle_lin = nn.Linear(cfg.graph_features.vehicle, self.cfg.vehicle_node_feature_size)

        if self.cfg.temporal.enabled:
            self.temporal_edge_encoder = Time2Vec(dim=self.cfg.temporal_time_to_vec_encoding_size)

        self.gnn_convs = nn.ModuleList()
        for i in range(self.cfg.gnn_conv_layers):
            in_channels_edge = {
                ("lanelet", "to", "lanelet"): cfg.lanelet_network_encoder.edge_feature_size,
                ("vehicle", "to", "vehicle"): cfg.graph_features.vehicle_to_vehicle,
                ("lanelet", "to", "vehicle"): cfg.graph_features.lanelet_to_vehicle,
                ("vehicle", "to", "lanelet"): cfg.graph_features.vehicle_to_lanelet,
            }
            if self.cfg.temporal.enabled:
                in_channels_edge[("vehicle", "temporal", "vehicle")] = self.cfg.temporal_time_to_vec_encoding_size
            conv = CustomHGTConv(
                in_channels_node={
                    "vehicle": self.cfg.vehicle_node_feature_size,
                    "lanelet": self.cfg.lanelet_node_feature_size,
                    # CustomHGTConv.GLOBAL_CONTEXT_NODE: self.cfg.global_context_size,
                },
                in_channels_edge=in_channels_edge,
                attention_channels=self.cfg.attention_channels,
                out_channels_node={
                    "vehicle": self.cfg.vehicle_node_feature_size,
                    "lanelet": self.cfg.lanelet_node_feature_size,
                    # CustomHGTConv.GLOBAL_CONTEXT_NODE: self.cfg.global_context_size,
                },
                # see HeteroData.metadata()
                metadata=(NODE_TYPES, EDGE_TYPES_TEMPORAL if self.cfg.temporal.enabled else EDGE_TYPES),
                attention_heads=self.cfg.gnn_attention_heads,
                neighbor_aggregation=self.cfg.gnn_neighbor_aggregation,
                aggregation=self.cfg.gnn_aggregation,
                add_self_loops=[
                    ("lanelet", "to", "lanelet"),
                    ("vehicle", "to", "vehicle"),
                ],
                global_context=False
            )
            self.gnn_convs.append(conv)

        self.batch_norm_v = BatchNorm(cfg.graph_features.vehicle) if self.cfg.batch_norm_input else nn.Identity()
        self.batch_norm_v2v = BatchNorm(cfg.graph_features.vehicle_to_vehicle) if self.cfg.batch_norm_input else nn.Identity()
        self.batch_norm_v2l = BatchNorm(cfg.graph_features.vehicle_to_lanelet) if self.cfg.batch_norm_input else nn.Identity()

    def forward(self, data: Union[CommonRoadData, CommonRoadDataTemporal]) -> Tensor:
        temporal_data = isinstance(data, CommonRoadDataTemporal)
        assert temporal_data or isinstance(data, CommonRoadData)

        node_type, edge_types = data.metadata()
        assert set(node_type) == set(NODE_TYPES) and \
            set(edge_types) == (set(EDGE_TYPES_TEMPORAL) if temporal_data else set(EDGE_TYPES))

        x_lanelet, edge_index_lanelet, edge_attr_lanelet = self.lanelet_encoder(data)

        x_vehicle = torch.cat([
            data.vehicle.velocity/10,
            data.vehicle.acceleration,
            data.vehicle.length/5,
            data.vehicle.width/2,
            data.vehicle.yaw_rate,
            #data.vehicle.has_adj_lane_left,
            #data.vehicle.has_adj_lane_right
            #1/data.vehicle.num_lanelet_assignments
        ], dim=-1)

        x_dict = {
            "lanelet": x_lanelet,
            "vehicle": self.batch_norm_v(x_vehicle)
        }

        edge_attr_v2v = torch.cat([
            torch.cos(data.v2v.rel_orientation_ego),
            torch.sin(data.v2v.rel_orientation_ego),
            data.v2v.rel_position_ego/100,
            1 - torch.clamp(data.v2v.distance_ego, max=30.0)/30,
            data.v2v.rel_velocity_ego/10,
            data.v2v.rel_acceleration_ego,
            1 - torch.clamp(data.v2v.lanelet_distance, max=30.0)/30 
        ], dim=-1)

        edge_attr_v2l = torch.cat([
            data.v2l.v2l_lanelet_arclength_rel,
            data.v2l.v2l_lanelet_arclength_abs/100,
            torch.cos(data.v2l.v2l_heading_error),
            torch.sin(data.v2l.v2l_heading_error),
            data.v2l.v2l_lanelet_lateral_error,
        ], dim=-1)

        # assert data.v2l.v2l_lanelet_arclength_rel.min().item() >= 0.0 and data.v2l.v2l_lanelet_arclength_rel.max().item() <= 1.0

        edge_attr_v2l_normed = self.batch_norm_v2l(edge_attr_v2l)

        edge_attr_dict = {
            ('vehicle', 'to', 'vehicle'): self.batch_norm_v2v(edge_attr_v2v),
            ('lanelet', 'to', 'lanelet'): edge_attr_lanelet,
            ('vehicle', 'to', 'lanelet'): edge_attr_v2l_normed,
            ('lanelet', 'to', 'vehicle'): edge_attr_v2l_normed,
        }       
        if isinstance(data, CommonRoadDataTemporal):
            temporal_edge_attr = self.temporal_edge_encoder(data.vehicle_temporal_vehicle.edge_attr)
            edge_attr_dict[('vehicle', 'temporal', 'vehicle')] = temporal_edge_attr

        edge_index_dict: Dict[EdgeType, Tensor] = data.collect("edge_index")
        edge_index_dict[('lanelet', 'to', 'lanelet')] = edge_index_lanelet

        # project vehicle features to vehicle_node_feature_size-dimensional space
        x_dict["vehicle"] = self.vehicle_lin(x_dict["vehicle"])

        # # debugging
        #torch.set_printoptions(precision=2, sci_mode=False)
        #ego_edge_index = data.v2l.edge_index[0, :] == torch.where(data.v.id == -1)[0]
        # print(edge_attr_v2l[ego_edge_index])
        # print()

        # graph convolutions
        for idx, conv in enumerate(self.gnn_convs):
            x_dict = conv(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

        return x_dict["vehicle"]
