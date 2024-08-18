from typing import Dict, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import EdgeType
from itertools import chain

from commonroad_geometric.common.config import Config
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.common.torch_utils.helpers import assert_size
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from projects.geometric_models.drivable_area.models.encoder.lanelet_network_encoder import LaneletNetworkEncoder
from projects.geometric_models.drivable_area.models.modules.custom_hgt_conv import CustomHGTConv
from projects.geometric_models.drivable_area.models.modules.custom_hgt_conv_v2 import CustomHGTConv2
from projects.geometric_models.drivable_area.models.modules.time2vec import Time2Vec
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import LeakyReLU
from torch_geometric.typing import NodeType, EdgeType


NODE_TYPES = [
    "vehicle",
    # "lanelet",
]
EDGE_TYPES = [
    ("vehicle", "to", "vehicle"),
    # ("lanelet", "to", "lanelet"),
    # ("vehicle", "to", "lanelet"),
    # ("lanelet", "to", "vehicle"),
]
EDGE_TYPES_TEMPORAL = EDGE_TYPES + [
    ("vehicle", "temporal", "vehicle"),
]


class ScenarioEncoderModel(nn.Module):
    """Produces node representations"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.traffic
        self.cfg_input_graph_features = cfg.input_graph_features

        # == Lanelet nodes and lanelet-to-lanelet edges ==
        self.lanelet_encoder = LaneletNetworkEncoder(cfg=cfg.lanelet_network_encoder)
        assert self.lanelet_encoder.output_feature_size == self.cfg.lanelet_node_feature_size

        self.vehicle_lin = nn.Linear(cfg.input_graph_features.vehicle, self.cfg.vehicle_node_feature_size)

        self.v2v_lin = nn.Linear(cfg.input_graph_features.vehicle_to_vehicle, self.cfg.v2v_edge_feature_size)
        self.v2l_lin = nn.Linear(cfg.input_graph_features.vehicle_to_lanelet, self.cfg.v2l_edge_feature_size)
        self.l2v_lin = nn.Linear(cfg.input_graph_features.lanelet_to_vehicle, self.cfg.l2v_edge_feature_size)

        self.temporal_edge_encoder = TemporalEdgeEncoder(
            time_feature_index=0,
            time_vec_size=self.cfg.temporal_edge_feature_size,
            freq_init_const=self.cfg.temporal_edge_freq_init_const,
        )

        node_types = NODE_TYPES.copy()
        edge_types = EDGE_TYPES_TEMPORAL.copy() if self.cfg.temporal.enabled else EDGE_TYPES.copy()
        in_channels_edge = {
            # ("lanelet", "to", "lanelet"): cfg.lanelet_network_encoder.edge_feature_size,
            ("vehicle", "to", "vehicle"): self.cfg.v2v_edge_feature_size,
            # ("lanelet", "to", "vehicle"): self.cfg.l2v_edge_feature_size,
            # ("vehicle", "to", "lanelet"): self.cfg.v2l_edge_feature_size,
        }
        if self.cfg.temporal.enabled:
            in_channels_edge[("vehicle", "temporal", "vehicle")] = cfg.input_graph_features.vehicle_temporal_vehicle - 1 + self.cfg.temporal_edge_feature_size

        if "ablation" in self.cfg:  # TODO REMOVE
            if self.cfg.ablation.remove_l_v_edges:
                del in_channels_edge[("vehicle", "to", "lanelet")]
                del in_channels_edge[("lanelet", "to", "vehicle")]
                edge_types.remove(("vehicle", "to", "lanelet"))
                edge_types.remove(("lanelet", "to", "vehicle"))
            if self.cfg.ablation.remove_vtv_edges:
                del in_channels_edge[("vehicle", "temporal", "vehicle")]
                edge_types.remove(("vehicle", "temporal", "vehicle"))

        self.gnn_convs = nn.ModuleList()
        for i in range(self.cfg.gnn.conv_layers):
            in_channels_node = {
                "vehicle": self.cfg.vehicle_node_feature_size,
                # "lanelet": self.cfg.lanelet_node_feature_size,
            }
            out_channels_node = {
                "vehicle": self.cfg.vehicle_node_feature_size,
                # "lanelet": self.cfg.lanelet_node_feature_size,
            }
            if self.cfg.gnn.global_context:
                in_channels_node[CustomHGTConv2.GLOBAL_CONTEXT_NODE] = self.cfg.gnn.global_context_size
                out_channels_node[CustomHGTConv2.GLOBAL_CONTEXT_NODE] = self.cfg.gnn.global_context_size
            add_self_loops = None
            if self.cfg.gnn.add_self_loops:
                add_self_loops = [
                    # ("lanelet", "to", "lanelet"),
                    ("vehicle", "to", "vehicle"),
                ]

            if self.cfg.gnn.activation_fn == "gelu":
                activation_fn = F.gelu
            elif self.cfg.gnn.activation_fn == "leakyrelu":
                activation_fn = LeakyReLU(negative_slope=0.01)
            else:
                raise ValueError(f"Unknown GNN activation function {self.cfg.gnn.activation_fn}")

            conv = CustomHGTConv2(
                in_channels_node=in_channels_node,
                in_channels_edge=in_channels_edge.copy(),
                attention_channels=self.cfg.gnn.attention_channels,
                out_channels_node=out_channels_node,
                # see HeteroData.metadata()
                metadata=(node_types, edge_types),
                attention_heads=self.cfg.gnn.attention_heads,
                activation_fn=activation_fn,
                add_self_loops=add_self_loops,
                enable_residual_connection=self.cfg.gnn.residual_connection,
                enable_residual_weights=False,
                enable_global_context=self.cfg.gnn.global_context,
            )
            self.gnn_convs.append(conv)

    @staticmethod
    def extract_lanelet_data(data: CommonRoadData) -> Data:
        lanelet_data = Data()
        for key, value in chain(data.lanelet.items(), data.lanelet_to_lanelet.items()):
            lanelet_data[key] = value
        return lanelet_data

    def forward(self, data: Union[CommonRoadData, CommonRoadDataTemporal]) -> dict[str, Tensor]:
        temporal_data = isinstance(data, CommonRoadDataTemporal) and data.vtv.num_edges > 0
        assert temporal_data or isinstance(data, CommonRoadData)

        

        # add multi-hop edges
        # if self.cfg.vehicle_lanelet_vehicle_edges:
        #     # vehicle -> lanelet -> vehicle
        #     # TODO doesn't really make sense
        #     data[("vehicle", "lanelet", "vehicle")].edge_index = transitive_edges(
        #         data.vehicle_to_lanelet.edge_index,
        #         data.lanelet_to_vehicle.edge_index,
        #     )
        #     num_vehicle_lanelet_vehicle_edges = data[("vehicle", "lanelet", "vehicle")].edge_index.size(1)
        #     data[("vehicle", "lanelet", "vehicle")].edge_attr = torch.zeros(
        #         (num_vehicle_lanelet_vehicle_edges, 0),
        #         dtype=torch.float32,
        #     )

        x_vehicle = torch.cat([
            data.vehicle.velocity/10,
            data.vehicle.acceleration,
            data.vehicle.length/5,
            data.vehicle.width/2,
            # data.vehicle.yaw_rate,
            # data.vehicle.lanelet_lateral_error,
            # data.vehicle.heading_error,
            data.vehicle.has_adj_lane_left,
            data.vehicle.has_adj_lane_right
        ], dim=-1)
        assert_size(x_vehicle, (None, self.cfg_input_graph_features.vehicle))

        edge_index_dict: dict[EdgeType, Tensor] = data.collect("edge_index")
        # edge_attr_dict: dict[EdgeType, Tensor] = data.collect("edge_attr")

        # compute initial lanelet features
        lanelet_data = self.extract_lanelet_data(data)

        if "relative_vertices" in lanelet_data:
            x_lanelet, edge_attr_lanelet = self.lanelet_encoder(lanelet_data)
            # TODO

        x_dict = {
            # "lanelet": x_lanelet,
            "vehicle": x_vehicle
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

        # edge_attr_v2l = torch.cat([
        #     data.v2l.v2l_lanelet_arclength_rel,
        #     data.v2l.v2l_lanelet_arclength_abs/100,
        #     torch.cos(data.v2l.v2l_heading_error),
        #     torch.sin(data.v2l.v2l_heading_error),
        #     data.v2l.v2l_lanelet_lateral_error,
        # ], dim=-1)

        edge_attr_dict = {
            ('vehicle', 'to', 'vehicle'): edge_attr_v2v,
            # ('lanelet', 'to', 'lanelet'): edge_attr_lanelet,
            # ('vehicle', 'to', 'lanelet'): edge_attr_v2l,
            # ('lanelet', 'to', 'vehicle'): None,
        }  

        # compute temporal edge features
        if temporal_data:
            edge_attr_dict["vehicle", "temporal", "vehicle"] = self.temporal_edge_encoder(data.vtv.edge_attr)

        # project vehicle features to vehicle_node_feature_size-dimensional space
        x_dict["vehicle"] = self.vehicle_lin(x_dict["vehicle"])
        edge_attr_dict[("vehicle", "to", "vehicle")] = self.v2v_lin(edge_attr_dict[("vehicle", "to", "vehicle")])
        # edge_attr_dict[("vehicle", "to", "lanelet")] = self.v2l_lin(edge_attr_dict[("vehicle", "to", "lanelet")])
        # if self.cfg_input_graph_features.lanelet_to_vehicle == 0:
        #     edge_attr_dict[("lanelet", "to", "vehicle")] = torch.zeros((edge_attr_dict[("lanelet", "to", "vehicle")].size(0), self.cfg.l2v_edge_feature_size), device=edge_attr_dict[("lanelet", "to", "vehicle")].device)
        # else:
        #     edge_attr_dict[("lanelet", "to", "vehicle")] = self.l2v_lin(edge_attr_dict[("lanelet", "to", "vehicle")])

        # del x_dict['lanelet']
        del edge_index_dict[("vehicle", "to", "lanelet")]
        del edge_index_dict[("lanelet", "to", "vehicle")]
        del edge_index_dict[("lanelet", "to", "lanelet")]
        # del edge_attr_dict[("vehicle", "to", "lanelet")]
        # del edge_attr_dict[("lanelet", "to", "vehicle")]
        # del edge_attr_dict[("lanelet", "to", "lanelet")]

        if "ablation" in self.cfg:  # TODO REMOVE
            if self.cfg.ablation.remove_l_v_edges:
                del edge_index_dict[("vehicle", "to", "lanelet")]
                del edge_index_dict[("lanelet", "to", "vehicle")]
                del edge_attr_dict[("vehicle", "to", "lanelet")]
                del edge_attr_dict[("lanelet", "to", "vehicle")]
            if self.cfg.ablation.remove_vtv_edges:
                del edge_index_dict[("vehicle", "temporal", "vehicle")]
                del edge_attr_dict[("vehicle", "temporal", "vehicle")]

            if self.cfg.ablation.get("remove_edge_attributes", default=False):
                edge_attr_dict = {
                    edge_type: torch.zeros_like(attr)
                    for edge_type, attr in edge_attr_dict.items()
                }

        # graph convolutions
        for conv in self.gnn_convs:
            x_dict = conv(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

        # x_vehicle = torch.relu(x_dict['vehicle']) # TODO
        x_vehicle = x_dict['vehicle']

        return x_vehicle


class TemporalEdgeEncoder(nn.Module):

    def __init__(self, time_feature_index: int, time_vec_size: int, freq_init_const: float):
        super().__init__()
        assert time_feature_index == 0
        self.time_feature_index = 0
        self.time2vec = Time2Vec(dim=time_vec_size, freq_init_const=freq_init_const)

    def forward(self, x: Tensor) -> Tensor:
        time_vec = self.time2vec(x[..., self.time_feature_index].unsqueeze(-1))
        return torch.cat([
            # x[..., :self.time_feature_index],  empty because time_feature_index = 0
            time_vec,
            x[..., self.time_feature_index + 1:],
        ], dim=-1)
