from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import Tanh
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.nn.models import MLP

from commonroad_geometric.common.torch_utils.pygeo import softmax
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.v2l_encoder import V2LEncoder, V2LEncoderConfig


class V2LGlobalEncoder(V2LEncoder):
    NUM_CONTEXT_FEATURES = 6 + V2LEncoder.INPUT_SIZE_L

    def __init__(
        self,
        output_size: int,
        offset_conditioning: bool = False,
        velocity_conditioning: bool = False,
        config: Optional[V2LEncoderConfig] = None,
        **kwargs: Any
    ):
        self.global_output_size = output_size
        config = config or V2LEncoderConfig(**kwargs)
        config.mlp_layers_embed = 1
        super().__init__(
            output_size=config.hidden_size,
            offset_conditioning=offset_conditioning,
            velocity_conditioning=velocity_conditioning,
            config=config
        )

    def reset_config(self) -> None:
        self.config = V2LEncoderConfig()  # TODO

    def build(
        self,
        data: Optional[CommonRoadData],
        trial = None,
        enable_batch_norm: Optional[bool] = None
    ) -> None:
        super().build(data)

        if enable_batch_norm is None:
            if data is not None:
                batch_size = data.lanelet.batch.max().item() + 1 if hasattr(data.lanelet, "batch") else 1
                enable_batch_norm = batch_size > 1
            else:
                enable_batch_norm = False
        # self.pooling_layer = SAGPooling(
        #     in_channels=self.output_size,
        #     ratio=1,
        # )
        dim = self.global_output_size
        # self.key_feature_encoder = FourierFeatures(
        #     num_input_channels=2,
        #     mapping_size=dim,
        #     scale=1.0,
        # )
        self.global_lin_key = MLP(
            in_channels=V2LGlobalEncoder.NUM_CONTEXT_FEATURES,
            out_channels=self.config.hidden_size,
            num_layers=2,
            hidden_channels=self.config.hidden_size,
            # norm="batch_norm" if enable_batch_norm else None,
            act=nn.Tanh()
        )
        self.global_key_act = nn.Sigmoid()

        # not used
        self.global_lin_reconstruct = nn.Linear(V2LGlobalEncoder.NUM_CONTEXT_FEATURES, self.config.hidden_size)
        self.global_reconstruct_act = nn.Tanh()

        # not used
        self.global_msg_encoder = MLP(
            in_channels=self.config.hidden_size + V2LGlobalEncoder.NUM_CONTEXT_FEATURES,
            out_channels=self.config.hidden_size,
            num_layers=2,
            hidden_channels=self.config.hidden_size,
            # norm="batch_norm" if enable_batch_norm else None,
            act=Tanh
        )
        # self.global_msg_norm = nn.Identity(dim)
        self.global_msg_norm = nn.Identity()  # BatchNorm(128)
        self.global_msg_act = nn.Tanh()
        # self.global_pooler = GlobalAttention(
        #     gate_nn=MLP(
        #         input_size=dim,
        #         output_size=dim,
        #         net_arch=dim,
        #     ),
        #     nn=MLP(
        #         input_size=dim,
        #         output_size=dim,
        #         net_arch=dim,
        #     )
        # )
        # self.global_pooler =
        # self.global_pooler = Set2Set(
        #     in_channels=dim,
        #     processing_steps=6,
        #     num_layers=1
        # )

        # used
        self.global_out_act = nn.Tanh()
        # self.global_out_norm = nn.Identity(dim)
        self.global_out_norm = nn.Identity()  # BatchNorm(128)

        if not hasattr(self, 'velocity_conditioning'):
            self.velocity_conditioning = False
        # used
        self.global_embed_mlp = MLP(
            # 2*dim if isinstance(self.global_pooler, Set2Set) else dim,
            in_channels=self.config.hidden_size + int(self.velocity_conditioning),
            hidden_channels=self.config.hidden_size,
            out_channels=dim,
            num_layers=2,
            norm=None,
            # norm="batch_norm" if enable_batch_norm else None,  # watch out for low batch size
            act=nn.Tanh()
        )

        # self.attention = nn.MultiheadAttention(
        #     embed_dim=self.output_size,
        #     num_heads=1,
        #     kdim=3
        # )

    def reset_parameters(self):
        super(V2LEncoder, self).reset_parameters()
        # nn.init.xavier_normal_(self.global_lin_key.weight)
        # nn.init.xavier_normal_(self.lin_out.weight, gain=1.0)
        return

    def forward(self, data: CommonRoadData) -> Tensor:
        batch_size = data.lanelet.batch.max().item() + 1
        z_r, x_v, x_l, edge_attr_l2l, edge_attr_v2l, message_intensities = self._compute_encodings(data)

        z_r_selected = z_r.index_select(0, data.walks)

        # edge_index, edge_attr, edge_mask = subgraph(
        #     data.walks,
        #     data.lanelet_to_lanelet.edge_index,
        #     data.lanelet_to_lanelet.edge_attr,
        #     relabel_nodes=True,
        #     return_edge_mask=True
        # )
        batch = data.lanelet.batch[data.walks]
        length = data.lanelet.length[data.walks]
        x_l_walks = x_l[data.walks]

        key_features = torch.column_stack([
            1 - data.integration_lower_limits,
            data.integration_upper_limits,
            data.integration_upper_limits - data.integration_lower_limits,
            (data.integration_upper_limits_abs - data.integration_lower_limits_abs) / data.path_length,
            data.cumulative_prior_length,
            length / data.path_length,
            x_l_walks
        ])
        global_lin_key = self.global_lin_key(key_features)
        key = softmax(global_lin_key, batch)  # self.global_key_act(global_lin_key)
        # key = self.global_key_act(global_lin_key)
        # reconstruct_features = torch.column_stack([
        #     data.cumulative_prior_length,
        #     data.integration_lower_limits,
        #     data.integration_upper_limits
        # ])
        # global_lin_reconstruct = self.global_lin_reconstruct(reconstruct_features)
        # reconstruct = self.global_reconstruct_act(global_lin_reconstruct)

        # msg_input = torch.cat([key*z_l_selected, key_features], dim=-1)
        msg_input = key * z_r_selected
        # msg_input = torch.cat([z_l_selected, key_features], dim=-1)
        msg = msg_input  # self.global_msg_encoder(msg_input)
        # msg_normed = self.global_msg_norm(msg)
        # global_msg_act = self.global_msg_act(msg)
        z_pooled = global_add_pool(msg, batch) / 3  # TODO

        # x, edge_index, _, batch, perm, score = self.pooling_layer(
        #     x=x,
        #     edge_index=edge_index,
        #     edge_attr=edge_attr,
        #     batch=batch,
        #     attn=key
        # )

        out_z_norm = self.global_out_norm(z_pooled) if batch_size > 1 else z_pooled
        out_z_act = self.global_out_act(out_z_norm)

        if hasattr(self, 'velocity_conditioning') and self.velocity_conditioning:
            if data.walk_velocity.ndim == 1:
                data.walk_velocity = data.walk_velocity.unsqueeze(-1)
            global_embed_mlp_in = torch.cat([
                out_z_act, data.walk_velocity / 10.0
            ], dim=-1)
        else:
            global_embed_mlp_in = out_z_act

        z_ego_route = self.global_embed_mlp(global_embed_mlp_in)

        assert z_ego_route.shape[0] == batch_size

        return z_ego_route, z_r, message_intensities  # TODO cleanup
