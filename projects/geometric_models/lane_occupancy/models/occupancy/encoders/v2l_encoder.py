from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Type

import torch
from optuna import Trial
from torch import Tensor, nn
from torch.nn import Tanh
from torch_geometric.nn import BatchNorm, HeteroConv
from torch_geometric.nn.models import MLP
from torch_geometric.utils.loop import add_self_loops

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.base_encoder import BaseOccupancyEncoder, BaseOccupancyEncoderConfig
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.conv_layers import L2LGNN, V2LConvLayer


@dataclass
class V2LEncoderConfig(BaseOccupancyEncoderConfig):
    activation_cls: Type[nn.Module] = Tanh
    batch_norm: bool = True
    dropout_prob: float = 0.0
    encoding_bias: bool = True
    hidden_size: int = 256
    intermediate_act: bool = True
    intermediate_norm: bool = True
    l2l_aggr: Literal['add', 'max'] = 'max'
    lanelet_edge_type_embedding_dim: int = 64
    l2l_interaction_layers: int = 4
    l2l_post_operation: Literal['sum', 'embed', 'none'] = 'none'
    l2l_self_loops: bool = True
    mlp_hidden_size_embed: int = 128
    mlp_hidden_size_l2l: int = 256
    mlp_hidden_size_v2l: int = 256
    mlp_layers_embed: int = 2
    mlp_layers_l2l: int = 1
    mlp_layers_v2l: int = 1
    v2l_aggr: Literal['add', 'max'] = 'max'


class V2LEncoder(BaseOccupancyEncoder):
    INPUT_SIZE_V = 8
    INPUT_SIZE_L = 6
    INPUT_SIZE_V2L = 5
    INPUT_SIZE_L2L = 10

    def __init__(
        self,
        output_size: int,
        offset_conditioning: bool = False,
        velocity_conditioning: bool = False, # TODO either remove or state unfinished features
        config: Optional[V2LEncoderConfig] = None,
        **kwargs: Any
    ):
        config = config or V2LEncoderConfig(**kwargs)
        self.config = config
        super().__init__(output_size=output_size, offset_conditioning=offset_conditioning, velocity_conditioning=velocity_conditioning)
        self.input_size_l = V2LEncoder.INPUT_SIZE_L + int(self.offset_conditioning)

    def reset_config(self) -> None:
        self.config = V2LEncoderConfig() # TODO

    def build(
        self,
        data: Optional[CommonRoadData],
        trial: Optional[Trial] = None
    ) -> None:
        self.config = V2LEncoderConfig() # TODO

        self.v2l = HeteroConv(convs={
            ('vehicle', 'to', 'lanelet'): V2LConvLayer(
                in_channels=V2LEncoder.INPUT_SIZE_V + self.input_size_l + V2LEncoder.INPUT_SIZE_V2L,
                hidden_channels=self.config.mlp_hidden_size_v2l,
                out_channels=self.config.hidden_size - self.input_size_l,
                num_layers=self.config.mlp_layers_v2l,
                aggr=self.config.v2l_aggr,
                #norm=self.config.batch_norm,
                dropout=self.config.dropout_prob,
                act=self.config.activation_cls
            )
        })

        self.v2l_act = self.config.activation_cls() if self.config.intermediate_act else nn.Identity()
        self.v2l_norm = BatchNorm(self.config.hidden_size - self.input_size_l) if self.config.intermediate_norm else nn.Identity()

        if self.config.l2l_interaction_layers:
            # self.l2l = L2LConvLayer(
            #     input_size = 2*self.config.hidden_size + 2*self.input_size_l + V2LEncoder.INPUT_SIZE_L2L + self.config.lanelet_edge_type_embedding_dim,
            #     output_size=self.config.hidden_size,
            #     hidden_size=self.config.mlp_layers_l2l*[self.config.hidden_size],
                
            # )
            self.l2l = L2LGNN(
                in_channels=self.config.hidden_size,
                hidden_channels=self.config.hidden_size,
                out_channels=self.config.hidden_size,
                num_layers=self.config.l2l_interaction_layers,
                edge_channels=V2LEncoder.INPUT_SIZE_L2L + self.config.lanelet_edge_type_embedding_dim,
                aggr=self.config.l2l_aggr,
                # norm=self.config.batch_norm,
                act=self.config.activation_cls(),
                mlp_hidden_channels=self.config.mlp_hidden_size_l2l,
                mlp_layers=self.config.mlp_layers_l2l
            )

        self.l2l_act = self.config.activation_cls() if self.config.intermediate_act else nn.Identity()
        self.l2l_norm = BatchNorm(self.config.hidden_size) if self.config.intermediate_norm else nn.Identity()

        self.embed_mlp = MLP(
            in_channels=self.config.hidden_size + self.input_size_l,
            hidden_channels=self.config.mlp_hidden_size_embed,
            out_channels=self.output_size,
            num_layers=self.config.mlp_layers_embed,
            #norm=self.config.batch_norm,
            dropout=self.config.dropout_prob,
            act=self.config.activation_cls
        )

        self.edge_types = [
            # LaneletEdgeType.PREDECESSOR,
            LaneletEdgeType.SUCCESSOR,
            LaneletEdgeType.ADJACENT_LEFT,
            #LaneletEdgeType.OPPOSITE_LEFT,
            LaneletEdgeType.ADJACENT_RIGHT,
            #LaneletEdgeType.OPPOSITE_RIGHT,
            # LaneletEdgeType.DIVERGING,
            LaneletEdgeType.MERGING,
            LaneletEdgeType.CONFLICTING,
            LaneletEdgeType.CONFLICT_LINK,
        ]
        self.edge_type_embedding = nn.Embedding(
            num_embeddings=len(self.edge_types),
            embedding_dim=self.config.lanelet_edge_type_embedding_dim
        )

        if self.config.l2l_post_operation == 'embed':
            self.emb_l2l_interaction_layers = nn.Linear(2*self.config.hidden_size, self.config.hidden_size)
            self.emb_l2l_interaction_layers_act = self.config.activation_cls()
        if self.config.encoding_bias:
            self.encoding_bias = nn.Parameter(torch.zeros(self.output_size))
        else:
            self.encoding_bias = 0

    def _compute_encodings(self, data: CommonRoadData) -> Tuple[Tensor, ...]:
        device = data.vehicle.velocity.device

        x_v = torch.cat([
            (data.vehicle.velocity-25)/10.0,
            (data.vehicle.acceleration-0.0)/1.0,
            (data.vehicle.length-5)/10.0,
            torch.log(torch.clamp(data.vehicle.velocity, 1e-3, 100.0)),
            data.vehicle.yaw_rate
            #1/data.vehicle.num_lanelet_assignments
        ], dim=-1)
        x_l_components = [
            data.lanelet.length/100.0,
            torch.log(data.lanelet.length/10.0),
            data.l.start_curvature*10.0,
            data.l.end_curvature*10.0,
            torch.abs(data.l.start_curvature)*10.0,
            torch.abs(data.l.end_curvature)*10.0
        ]
        if not hasattr(self, 'offset_conditioning'): # TODO: Delete
            self.offset_conditioning = False
        if self.offset_conditioning:
            if hasattr(data.lanelet, 'random_offsets'):
                x_l_components.append(data.lanelet.random_offsets / 10.0) 
            else:
                x_l_components.append(data.lanelet.length.new_zeros(data.lanelet.length.shape))
        x_l = torch.cat(x_l_components, dim=-1)
        
        edge_attr_v2l = torch.cat([
            2.0 * (data.v2l.v2l_lanelet_arclength_rel - 0.5),
            torch.cos(data.v2l.v2l_heading_error),
            torch.sin(data.v2l.v2l_heading_error),
            data.v2l.v2l_lanelet_lateral_error,
            torch.abs(data.v2l.v2l_lanelet_lateral_error),
        ], dim=-1)


        assert x_v.isfinite().all()
        assert x_l.isfinite().all()
        assert edge_attr_v2l.isfinite().all()

        sphi_v2l = self.v2l_act(self.v2l_norm(self.v2l(
            x_dict=dict(
                vehicle=x_v, 
                lanelet=x_l,
            ),
            edge_attr_dict={
                ('vehicle', 'to', 'lanelet'): edge_attr_v2l
            },
            edge_index_dict={
                ('vehicle', 'to', 'lanelet'): data.vehicle_to_lanelet.edge_index
            },
            dim_size_dict=dict(
                vehicle=data.vehicle.num_nodes,
                lanelet=data.lanelet.num_nodes
            )
        )['lanelet']))

        if self.config.l2l_interaction_layers:
            num_edges_l2l = data.l2l.edge_index.shape[1]
            edge_mask = torch.zeros(num_edges_l2l, dtype=torch.bool, device=device)
            edge_type_emb = torch.empty(
                num_edges_l2l,
                self.edge_type_embedding.embedding_dim,
                dtype=self.edge_type_embedding.weight.data.dtype,
                device=device,
            )
            for value, edge_type in enumerate(self.edge_types):
                edge_type_mask: Tensor = data.l2l.type.squeeze(1) == edge_type
                edge_mask.logical_or_(edge_type_mask)
                edge_type_ind = edge_type_mask.nonzero()
                edge_type_emb[edge_type_ind] = self.edge_type_embedding(
                    torch.empty_like(edge_type_ind, device=device).fill_(value),
                )

            edge_index_l2l = data.l2l.edge_index[:, edge_mask]

            edge_attr_l2l_all = torch.cat([
                torch.cos(data.l2l.relative_intersect_angle),
                torch.sin(data.l2l.relative_intersect_angle),
                torch.cos(data.l2l.relative_start_angle),
                torch.sin(data.l2l.relative_start_angle),
                torch.cos(data.l2l.relative_end_angle),
                torch.log2(data.l2l.relative_source_length),
                data.l2l.target_curvature*10,
                data.l2l.source_curvature*10,
                torch.abs(data.l2l.target_curvature)*10,
                torch.abs(data.l2l.source_curvature)*10
            ], dim=-1)
            assert edge_attr_l2l_all.isfinite().all()

            edge_attr_l2l = torch.hstack([
                edge_type_emb[edge_mask],
                edge_attr_l2l_all[edge_mask],
            ])

            z_l2l = torch.cat([
                sphi_v2l,
                x_l
            ], dim=-1)

            if self.config.l2l_self_loops:
                edge_index_l2l, edge_attr_l2l = add_self_loops(edge_index_l2l, edge_attr_l2l, fill_value=0.0)

            sphi2_l2l = self.l2l( # TODO: self-loops
                x=z_l2l, 
                edge_attr=edge_attr_l2l, 
                edge_index=edge_index_l2l
            )

            if not self.training:
                message_intensities = torch.column_stack([self.l2l.convs[i].message_intensities for i in range(len(self.l2l.convs))])
            else:
                message_intensities = None

            if self.config.l2l_post_operation == 'sum':
                z_r = sphi_v2l + sphi2_l2l
            elif self.config.l2l_post_operation == 'embed':
                z_r = self.emb_l2l_interaction_layers_act(self.emb_l2l_interaction_layers(
                    torch.cat([
                        sphi_v2l,
                        x_l,
                        sphi2_l2l
                    ], dim=-1)
                ))
            else:
                z_r = sphi2_l2l

        else:
            z_r = sphi_v2l

        return z_r, x_v, x_l, edge_attr_l2l, edge_attr_v2l, message_intensities

    def forward(self, data: CommonRoadData) -> Tensor:  
        z_r, x_v, x_l, edge_attr_l2l, edge_attr_v2l, message_intensities = self._compute_encodings(data)

        if self.embed_mlp.channel_list[0] == self.config.hidden_size:
            # hack for backwards comp.
            z_l = self.l2l_act(self.l2l_norm(z_r))
        else:
            z_l = torch.cat([
                self.l2l_act(self.l2l_norm(z_r)),
                x_l
            ], dim=-1)
        z = self.embed_mlp.forward(z_l) + self.encoding_bias

        if not z.isfinite().all():
            print("WARNING: not z.isfinite().all():")
            z = torch.nan_to_num(z)

        return z
