import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from optuna import Trial
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from commonroad_geometric.common.torch_utils.pygeo import get_batch_internal_indices
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.learning.training.optimizer.hyperparameter_optimizer_service import BaseOptimizerService
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin
from commonroad_geometric.rendering.plugins.render_traffic_graph_plugin import RenderTrafficGraphPlugin
from projects.geometric_models.lane_occupancy.models.occupancy.decoders.base_occupancy_decoder import BaseOccupancyDecoder, BaseOccupancyDecoderConfig
from projects.geometric_models.lane_occupancy.models.occupancy.decoders.ghost_decoder import GhostOccupancyDecoder
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.base_encoder import BaseOccupancyEncoder, BaseOccupancyEncoderConfig
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.v2l_encoder import V2LEncoder
from projects.geometric_models.lane_occupancy.models.occupancy.encoders.v2l_global_encoder import V2LGlobalEncoder
from projects.geometric_models.lane_occupancy.utils.preprocessing import preprocess_conditioning
from projects.geometric_models.lane_occupancy.utils.renderer_plugins import RenderLaneletOccupancyPredictionPlugin, RenderPathOccupancyPredictionPlugin

logger = logging.getLogger(__name__)

DEFAULT_OCCUPANCY_ENCODING_DIM = 32
DEFAULT_TIME_HORIZON = 60
# DEFAULT_DECODER_CLS = MLPOccupancyDecoder
DEFAULT_DECODER_CLS = GhostOccupancyDecoder
DEFAULT_PATH_LENGTH = 45.0
MIN_INTEGRAL_LENGTH = 2e-3

@dataclass
class OccupancyLossConfig:
    density_weighting: bool = False
    kld_weight: float = 0.02
    # sparsity_penalty_coef: float = 0.0 #0.02
    # beta_penalty_coef: float = 0.0 #10.0
    l1_penalty_coef: float = 0.01
    l2_penalty_coef: float = 0.0
    log_likelihood_eps: float = 1e-5
    min_relevance_weight: float = 0.2
    normalize_integrals: bool = True
    path_auxillary_loss: bool = False
    time_discount_factor: float = 0.99
    positive_loss_weight: float = 0.5
    integral_resolution: int = 20


@dataclass
class OptimizerConfig:
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    momentum: float = 0.9
    learning_rate: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0


@dataclass
class OccupancyModelConfig:
    decoder_cls: Type[BaseOccupancyDecoder] = DEFAULT_DECODER_CLS
    decoder_config: Optional[BaseOccupancyDecoderConfig] = None
    dt: float = 0.04
    encoder_cls: Optional[Type[BaseOccupancyEncoder]] = None
    encoder_config: Optional[BaseOccupancyEncoderConfig] = None
    encoding_size: int = DEFAULT_OCCUPANCY_ENCODING_DIM
    offset_conditioning: bool = False
    velocity_conditioning: bool = False
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    offset_std: float = 100.0
    path_conditioning: bool = True
    path_length: float = DEFAULT_PATH_LENGTH
    reload_path_testing_freq: int = 75
    time_horizon: int = DEFAULT_TIME_HORIZON
    variational: bool = False
    loss: Optional[OccupancyLossConfig] = None


class OccupancyModel(BaseGeometric):

    def __init__(
        self,
        config: Optional[OccupancyModelConfig] = None,
    ):
        config = OccupancyModelConfig(dict(config)) # TODO
        super(OccupancyModel, self).__init__(None)
        self.config = config
        self.config.decoder_cls = DEFAULT_DECODER_CLS

        if self.config.loss is None:
            self.config.loss = OccupancyLossConfig()

        if self.config.encoder_cls is not None:
            encoder_cls = self.config.encoder_cls
        elif config.path_conditioning:
            encoder_cls = V2LGlobalEncoder
        else:
            encoder_cls = V2LEncoder

        self.encoder = encoder_cls(
            output_size=self.config.encoding_size,
            velocity_conditioning=self.config.velocity_conditioning,
            offset_conditioning=self.config.offset_conditioning,
            config=config.encoder_config
        )
        self.decoder = self.config.decoder_cls(
            input_size=self.config.encoding_size,
            offset_conditioning=self.config.offset_conditioning,
            config=config.decoder_config
        )

        # for wandb export
        config.encoder_config = self.encoder.config
        config.decoder_config = self.decoder.config # TODO

        self._call_count: int = 0
        self._call_count_train: int = 0

        logger.info(f"Initialized OccupancyModel with encoder={type(self.encoder).__name__} and decoder={type(self.decoder).__name__}")

    def reset_config(self) -> None:
        self.config = OccupancyModelConfig() # TODO
        self.config.loss = OccupancyLossConfig()
        self.decoder.reset_config()
        self.encoder.reset_config()

    def reset_loss_config(self) -> None:
        self.config.loss = OccupancyLossConfig()

    def reset_optimizer_config(self) -> None:
        self.config.optimizer_config = OptimizerConfig()

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRendererPlugin]]:
        plugins = [
            RenderLaneletNetworkPlugin(),
            #RenderLaneletNetworkPlugin(render_id=True),
            RenderPathOccupancyPredictionPlugin(),
            RenderTrafficGraphPlugin(),
            RenderObstaclesPlugin(),
        ]
        return plugins

    def configure_optimizer(
        self,
        trial: Trial,
        optimizer_service: BaseOptimizerService,
    ) -> Optimizer:
        cfg = self.config.optimizer_config
        if cfg.optimizer is torch.optim.RMSprop:
            return torch.optim.RMSprop(
                self.parameters(), 
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer is torch.optim.SGD:
            return torch.optim.SGD(
                self.parameters(), 
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay
            )
        else:
            return torch.optim.Adam(
                self.parameters(), 
                lr=cfg.learning_rate,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay
            )

    def _build(self, data: CommonRoadData, trial: Trial) -> None:
        self.config = OccupancyModelConfig() # TODO
        if self.config.loss is None:
            self.config.loss = OccupancyLossConfig()

        time_discount_vec_abs = torch.cumprod(self.config.loss.time_discount_factor * data.v2v.edge_attr.new_ones((self.config.time_horizon,)), 0).unsqueeze(0)
        self.time_discount_vec = time_discount_vec_abs / time_discount_vec_abs.mean()

        self.encoder.build(data=data)
        if self.config.variational:
            self.fc_mu = nn.Linear(self.config.encoding_size, self.config.encoding_size)
            self.fc_var = nn.Linear(self.config.encoding_size, self.config.encoding_size)
        self.decoder.build(data=data)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def set_batch(
        self,
        data: CommonRoadData
    ) -> Tensor:
        device = data.lanelet_to_lanelet.edge_index.device
        if not hasattr(data.lanelet, 'batch'):
            data.lanelet.batch = torch.zeros((
                data.lanelet.num_nodes),
                dtype=torch.long,
                device=device
            )
            data.vehicle.batch = torch.zeros((
                data.vehicle.num_nodes),
                dtype=torch.long,
                device=device
            )

    def encode(self, data: CommonRoadData) -> Tensor:
        data.path_length = self.config.path_length
        if not self.training:
            self.set_batch(data)
        z = self.encoder(data)
        if self.config.variational:
            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            return mu, log_var
        else:
            return z

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        data: CommonRoadData,
        domain: Union[int, Tensor] = 300,
        validation: bool = False
    ) -> Tensor:
        device = data.lanelet_to_lanelet.edge_index.device
        if not self.training:
            self.set_batch(data)
            if self.config.path_conditioning:
                if not hasattr(data, 'walks'):
                    if not validation and hasattr(self, '_flattened_walks') and self._call_count % self.config.reload_path_testing_freq != 0:
                        data.walks = self._flattened_walks.to(device)
                        if self.config.velocity_conditioning:
                            data.walk_velocity = self._walk_velocity_flattened.to(device)
                        data.walk_masks = self._walk_mask_flattened.to(device)
                        data.cumulative_prior_length = self._cumulative_prior_length_flattened.to(device)
                        data.integration_lower_limits = self._integration_lower_limits_flattened.to(device)
                        data.integration_upper_limits = self._integration_upper_limits_flattened.to(device)
                        data.cumulative_prior_length_abs = self._cumulative_prior_length_flattened_abs.to(device)
                        data.integration_lower_limits_abs = self._integration_lower_limits_flattened_abs.to(device)
                        data.integration_upper_limits_abs = self._integration_upper_limits_flattened_abs.to(device)
                    else:
                        data = self.train_preprocess(data)
                if data.walks.max().item() >= data.lanelet.num_nodes:
                    data = self.train_preprocess(data)

        self._call_count += 1
        if self.training:
            self._call_count_train += 1

        data.path_length = self.config.path_length
        z = self.encoder(data)

        encoding = self.encode(data)
        if self.config.variational:
            mu, log_var = encoding
            z = self.reparameterize(mu, log_var)
        else:
            z = encoding

        if not self.training:

            if not validation:
                # if not hasattr(data.lanelet, 'occupancy_time_horizon'):
                #     from commonroad_geometric.dataset.postprocessing.implementations.lanelet_occupancy_post_processor import LaneletOccupancyPostProcessor
                #     post_processor = LaneletOccupancyPostProcessor(
                #         time_horizon=1,
                #         discretization_resolution=self.config.loss.integral_resolution
                #     )
                #     data = post_processor([data])[0]
                # time_horizon = data.lanelet.occupancy_time_horizon if isinstance(data.lanelet.occupancy_time_horizon, int) else data.lanelet.occupancy_time_horizon[0].item()
                time_horizon = self.config.time_horizon

                if self.config.path_conditioning:
                    z_ego_route = z[0]
                    y_path_conditioning = self.decoder(
                        domain=domain,
                        z=z_ego_route,
                        lanelet_length=self.config.path_length,
                        dt=self.config.dt,
                        time_horizon=time_horizon
                    )
                    return z, y_path_conditioning
                else:
                    z_ego_route = z
                    y_lanelets = self.decoder(
                        domain=domain,
                        z=z,
                        lanelet_length=data.lanelet.length,
                        dt=self.config.dt,
                        time_horizon=time_horizon
                    )
                    if self.config.variational:
                        return z, mu, log_var, y_lanelets
                    else:
                        return z, y_lanelets

        if self.config.variational:
            return z, mu, log_var
        else:
            return (z,)

    def preprocess_conditioning(
        self,
        data: CommonRoadData,
        walks: Tensor,
        walk_start_length: Optional[Tensor] = None,
        walk_velocity: Optional[Tensor] = None,
        walk_masks: Optional[Tensor] = None
    ) -> None:
        self.set_batch(data)

        preprocess_conditioning(
            data=data,
            walks=walks,
            walk_start_length=walk_start_length,
            walk_velocity=walk_velocity,
            path_length=self.config.path_length,
            walk_masks=walk_masks
        )

        # data.walks = flattened_walks
        # data.walk_masks = walk_mask_flattened
        # data.cumulative_prior_length = cumulative_prior_length_flattened
        # data.integration_lower_limits = integration_lower_limits_flattened
        # data.integration_upper_limits = integration_upper_limits_flattened
        # data.cumulative_prior_length_abs = cumulative_prior_length_flattened_abs
        # data.integration_lower_limits_abs = integration_lower_limits_flattened_abs
        # data.integration_upper_limits_abs = integration_upper_limits_flattened_abs

        self._walk_mask_flattened = data.walk_masks
        self._flattened_walks = data.walks
        self._cumulative_prior_length_flattened = data.cumulative_prior_length
        self._integration_lower_limits_flattened = data.integration_lower_limits
        self._integration_upper_limits_flattened = data.integration_upper_limits
        self._cumulative_prior_length_flattened_abs = data.cumulative_prior_length_abs
        self._integration_lower_limits_flattened_abs = data.integration_lower_limits_abs
        self._integration_upper_limits_flattened_abs = data.integration_upper_limits_abs

    def train_preprocess(
        self,
        data: CommonRoadData
    ) -> CommonRoadData:
        """Allows custom pre-processing of data instances during training.
        A possible application could be to apply Gaussian noise.

        Args:
            data (CommonRoadData): Incoming Data instance.

        Returns:
            CommonRoadData: Pre-processed Data instance.
        """
        device = data.lanelet_to_lanelet.edge_index.device

        if self.config.offset_conditioning:
            data.lanelet.random_offsets = torch.normal(
                torch.zeros((data.l.num_nodes, ), device=device),
                torch.full((data.l.num_nodes, ), self.config.offset_std, device=device) / data.l.length.squeeze(-1)
            ).unsqueeze(-1)

            max_vehicle_count = data.lanelet.occupancy_max_vehicle_count if isinstance(data.lanelet.occupancy_max_vehicle_count, int) else data.lanelet.occupancy_max_vehicle_count[0].item()
            time_horizon = data.lanelet.occupancy_time_horizon if isinstance(data.lanelet.occupancy_time_horizon, int) else data.lanelet.occupancy_time_horizon[0].item()

            # BATCH_SIZE x TIME_HORIZON x N_SLOTS x 3
            y_cont : Tensor = data.lanelet.occupancy_continuous.view(
                data.lanelet.occupancy_continuous.shape[0], time_horizon, max_vehicle_count, 3
            )
            y_cont[..., :2] -= data.lanelet.random_offsets[:, None, None]
            data.lanelet.occupancy_continuous = y_cont.flatten(1)

            return data

        if not self.config.path_conditioning:
            return data

        # TODO: consider lanelet lengths instead of just assuming all to be equal length.
        # let trajectory length in meters be a parameter.


        random_walk = torch.ops.torch_cluster.random_walk
        from torch_sparse import SparseTensor
        from commonroad_geometric.common.torch_utils.pygeo import get_batch_sizes, get_batch_start_indices
        from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType

        #edge_filter = data.lanelet_to_lanelet.traffic_flow
        edge_filter = data.lanelet_to_lanelet.type == LaneletEdgeType.SUCCESSOR

        edge_index = data.lanelet_to_lanelet.edge_index[:, edge_filter.squeeze(1)]
        num_nodes = data.lanelet.num_nodes
        batch_size = data.lanelet.batch.max().item() + 1
        from torch_geometric.utils.num_nodes import maybe_num_nodes
        N = maybe_num_nodes(edge_index, num_nodes)
        node_idx = torch.arange(num_nodes, device=edge_index.device)
        self.walks_per_node = 1
        self.walk_length = 3
        self.n_walk_nodes = self.walk_length + 1
        #if self.config.path_length is not None:
            # assert self.config.path_length < self.n_walk_nodes
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj  # TODO why...
        batch = node_idx.repeat_interleave(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        self.p = 1.0
        self.context_size = 1
        self.q = 1.0
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        rw_directed = torch.stack([row for row in rw if row.unique(return_counts=True)[1].max().item() == 1 ])

        rw_start_nodes = rw_directed[:, 0]
        rw_batch = data.lanelet.batch.index_select(0, rw_start_nodes)
        rw_batch_sizes = get_batch_sizes(rw_batch)
        rw_selections = (torch.rand(batch_size, device=device) * rw_batch_sizes).int()
        rw_batch_start_indices = get_batch_start_indices(rw_batch_sizes, is_batch_sizes=True)
        rw_batch_selections = rw_batch_start_indices + rw_selections
        walks = rw_directed[rw_batch_selections, :]

        #n_walks = walks.shape[0]

        # if self.config.path_length is None:
        #     lower_bounds = self.n_walk_nodes * torch.rand(n_walks, device=device)
        #     upper_bounds = self.n_walk_nodes - (self.n_walk_nodes - lower_bounds) * torch.rand(n_walks, device=device)
        # else:
        #     lower_bounds = (self.n_walk_nodes - self.config.path_length) * torch.rand(n_walks, device=device)
        #     upper_bounds = lower_bounds + self.config.path_length
        # bounds = torch.column_stack([lower_bounds, upper_bounds])
        #walk_indeces = torch.arange(self.n_walk_nodes, device=device)[None, :].repeat(n_walks, 1)
        #walk_masks = (lower_bounds.unsqueeze(-1) < walk_indeces + 1) & (walk_indeces < (upper_bounds.unsqueeze(-1)))
        self.preprocess_conditioning(data, walks)

        return data

    def compute_loss(
        self,
        out: Tensor,
        data: CommonRoadData,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any], Dict[str, Tensor]]:  
        if self.config.loss is None:
            self.config.loss = OccupancyLossConfig()

        device = data.v2v.edge_index.device

        if self.config.path_conditioning:
            assert isinstance(out, tuple)
            if isinstance(out[0], tuple):
                z_ego_route, z_lanelet  = out[0][:2]
            else:
                z_ego_route, z_lanelet  = out[:2]
        else:
            if self.config.variational:
                z_lanelet, mu, log_var = out[:3]
            else:
                z_lanelet = out[0]

        if not hasattr(data.lanelet, 'occupancy_max_vehicle_count'):
            from commonroad_geometric.dataset.postprocessing.implementations.lanelet_occupancy_post_processor import LaneletOccupancyPostProcessor
            post_processor = LaneletOccupancyPostProcessor(
                time_horizon=1,
                discretization_resolution=None
            )
            data = post_processor([data])[0]

        max_vehicle_count = data.lanelet.occupancy_max_vehicle_count if isinstance(data.lanelet.occupancy_max_vehicle_count, int) else data.lanelet.occupancy_max_vehicle_count[0].item()
        time_horizon = data.lanelet.occupancy_time_horizon if isinstance(data.lanelet.occupancy_time_horizon, int) else data.lanelet.occupancy_time_horizon[0].item()

        # BATCH_SIZE x TIME_HORIZON x N_SLOTS x 3
        y_cont : Tensor = data.lanelet.occupancy_continuous.view(
            data.lanelet.occupancy_continuous.shape[0], time_horizon, max_vehicle_count, 3
        )

        n_vehicle_slots = y_cont.shape[2]
        if self.config.path_conditioning:
            y_cont_path = y_cont[data.walks]
            y_cont_neg_path = y_cont_path.new_zeros((y_cont_path.shape[0], time_horizon, n_vehicle_slots + 1, 3))
            y_cont_neg_path[:, :, 0, 1] = 1.0
            y_cont_neg_path[:, :, 0, 2] = 1.0
            y_cont_neg_path[:, :, :-1, 0] = y_cont_path[:, :, :, 1]
            y_cont_neg_path[:, :, 1:, 1] = y_cont_path[:, :, :, 0]
            y_cont_neg_path[:, :, 1:, 2] = y_cont_path[:, :, :, 2]
        # out = out[data.walks]

        y_cont_neg = y_cont.new_zeros((y_cont.shape[0], time_horizon, n_vehicle_slots + 1, 3))
        y_cont_neg[:, :, 0, 1] = 1.0
        y_cont_neg[:, :, 0, 2] = 1.0
        y_cont_neg[:, :, :-1, 0] = y_cont[:, :, :, 1]
        y_cont_neg[:, :, 1:, 1] = y_cont[:, :, :, 0]
        y_cont_neg[:, :, 1:, 2] = y_cont[:, :, :, 2]

        
        def compute_integrals(
            y: Tensor,
            res: int,
            data: CommonRoadData,
            time_horizon: int,
            nprob: bool = False,
            path_conditioning: bool = False
        ) -> Tuple[Tensor, Dict[str, Tensor]]:

            if path_conditioning:
                lanelet_length = self.config.path_length
                batch_size = data.lanelet.batch.max().item() + 1
            else:
                lanelet_length = data.lanelet.length
                batch_size = data.lanelet.num_nodes

            device = y.device
            n_slots = y.shape[2]
            n_lanelets = y.shape[0]
            time_horizon = y.shape[1]

            # N_SLOTS x N_LANELETS x TIME_HORIZON x 3
            y_view = y.permute(2, 0, 1, 3)

            # N_SLOTS x N_LANELETS x TIME_HORIZON 
            bounds_low_raw = y_view[:, :, :, 0]
            bounds_high_raw = y_view[:, :, :, 1]
            mask_raw = y_view[:, :, :, 2].bool()

            if path_conditioning:
                # N_LANELETS
                path_lower_limits = data.integration_lower_limits
                path_upper_limits = data.integration_upper_limits
                cumulative_prior_length = data.cumulative_prior_length
                walk_batch = data.lanelet.batch.index_select(0, data.walks)
                    
                # N_SLOTS x N_LANELETS x TIME_HORIZON
                #bounds_offset =  (cumulative_prior_length - path_lower_limits)[None, :, None]
                #bounds_low = (bounds_low_raw + bounds_offset)
                #bounds_high = (bounds_high_raw + bounds_offset)

                # N_SLOTS x N_LANELETS x TIME_HORIZON
                walk_lanelet_lengths_flattened = data.lanelet.length[data.walks]

                if self.config.velocity_conditioning:
                    lanelet_rel_walk_velocity = data.walk_velocity[walk_batch] / walk_lanelet_lengths_flattened
                    time_vector = self.config.dt * torch.arange(time_horizon, device=data.device, dtype=torch.float32)
                    displacement_vector = lanelet_rel_walk_velocity.repeat(1, time_horizon)*time_vector[None, :].repeat(n_lanelets, 1)
                    bounds_low_raw -= displacement_vector
                    bounds_high_raw -= displacement_vector

                bounds_low_abs = bounds_low_raw * walk_lanelet_lengths_flattened
                bounds_high_abs = bounds_high_raw * walk_lanelet_lengths_flattened

                bounds_low_abs_offset = bounds_low_abs + data.cumulative_prior_length_abs[None, :, None] - data.integration_lower_limits_abs[None, :, None]
                bounds_high_abs_offset = bounds_high_abs + data.cumulative_prior_length_abs[None, :, None] - data.integration_lower_limits_abs[None, :, None]

                bounds_low_abs_offset = torch.clamp(bounds_low_abs_offset, min=0.0)
                bounds_high_abs_offset = torch.clamp(bounds_high_abs_offset, max=self.config.path_length)

                local_integration_lengths = bounds_high_abs_offset - bounds_low_abs_offset
                # local_integral_compute_mask = local_integration_lengths >= 0.02
                local_integration_domains = (bounds_low_abs_offset.unsqueeze(-1) + (local_integration_lengths.unsqueeze(-1))*\
                    torch.linspace(0, 1, self.config.loss.integral_resolution, device=device)[None, None, None, :]
                )

                walk_batch = data.lanelet.batch.index_select(0, data.walks)
                max_nodes = self.n_walk_nodes # get_batch_sizes(walk_batch).max().item()

                batch_internal_indices = get_batch_internal_indices(walk_batch)
                #max_nodes = get_batch_sizes(walk_batch).max().item()
                n_slots = y_view.shape[0]
                n_lanelets = len(walk_batch)

                mask_raw = mask_raw & (bounds_high_abs_offset >= 0) & (bounds_low_abs_offset <= self.config.path_length)

                #index = batch_indices[None, :, None, None].broadcast_to(src.shape).unsqueeze(1)
                # MAX_NODES x N_SLOTS x BATCH_SIZE x TIME_HORIZON x RESOLUTION 
                walk_lanelet_lengths_flattened = data.lanelet.length[data.walks]
                integration_domains = torch.zeros((
                    self.n_walk_nodes*n_slots,
                    batch_size,
                    time_horizon,
                    self.config.loss.integral_resolution
                ), dtype=torch.float32, device=device)
                mask_buffer = torch.zeros((
                    self.n_walk_nodes*n_slots,
                    batch_size,
                    time_horizon
                ), dtype=torch.bool, device=device)
                integration_lengths = torch.zeros((
                    self.n_walk_nodes*n_slots,
                    batch_size,
                    time_horizon
                ), dtype=torch.float32, device=device)
                for idx in range(n_lanelets):
                    start_slot_idx = batch_internal_indices[idx]*n_slots
                    end_slot_idx = (batch_internal_indices[idx] + 1)*n_slots
                    batch_idx = walk_batch[idx]
                    integration_domains[start_slot_idx:end_slot_idx, batch_idx, :, :] = local_integration_domains[:, idx, ...] / self.config.path_length
                    mask_buffer[start_slot_idx:end_slot_idx, batch_idx, :] = mask_raw[:, idx, ...]
                    integration_lengths[start_slot_idx:end_slot_idx, batch_idx, :] = local_integration_lengths[:, idx, ...] / self.config.path_length

                mask = mask_buffer & (integration_lengths >= MIN_INTEGRAL_LENGTH)


            else:
                bounds_low, bounds_high = bounds_low_raw, bounds_high_raw
                local_integration_lengths = bounds_high - bounds_low
                local_integral_compute_mask = local_integration_lengths >= MIN_INTEGRAL_LENGTH
                local_integration_domains = (bounds_low.unsqueeze(-1) + (local_integration_lengths.unsqueeze(-1))*\
                    torch.linspace(0, 1, res, device=device)[None, None, None, :]
                )
                local_mask = (mask_raw & local_integral_compute_mask)
                mask = local_mask
                integration_lengths = local_integration_lengths
                integration_domains = local_integration_domains
                max_nodes = 1

            # N_SLOTS x N_LANELETS x TIME_HORIZON x RESOLUTION
            # integration_domains

            z = z_ego_route if path_conditioning else z_lanelet

            # integration_domains = integration_domains[mask, :]
            # z = z[None, :, None, :].repeat(mask.shape[0], 1, mask.shape[2], 1)[mask, :]
            # lanelet_length = lanelet_length[None, :, None, :].repeat(mask.shape[0], 1, mask.shape[2], 1)[mask, :]

            occ_probs, compute_info = self.decoder.forward(
                domain=integration_domains,
                z=z,
                lanelet_length=lanelet_length,
                dt=self.config.dt,
                time_horizon=time_horizon
            )
            assert occ_probs.min() >= 0.0
            assert occ_probs.max() <= 1.0
            occ_probs = occ_probs[mask]


            if nprob:
                prob = 1 - occ_probs
            else:
                prob = occ_probs

            ll = torch.log(prob + self.config.loss.log_likelihood_eps)

            trapezoidal_integrals = torch.trapz(
                ll,
                x=integration_domains[mask]
            ) 
            if self.config.loss.normalize_integrals:
                trapezoidal_integrals /= integration_lengths[mask]
            trapezoidal_integrals_buffer = ll.new_zeros((
                n_slots*max_nodes, batch_size, time_horizon
            ))

            # (N_SLOTS*MAX_NODES) x BATCH_SIZE x TIME_HORIZON
            trapezoidal_integrals_buffer[mask] = trapezoidal_integrals
            
            # BATCH_SIZE x TIME_HORIZON
            integral = trapezoidal_integrals_buffer.sum(dim=0)

            #assert mask.any()

            return integral, compute_info, integration_domains[mask], occ_probs

        if not self.config.path_conditioning or self.config.loss.path_auxillary_loss:
            positive_integrals, compute_info_positive, integration_domains_positive, positive_occ_probs = compute_integrals(
                y_cont,
                res=self.config.loss.integral_resolution,
                data=data,
                time_horizon=time_horizon
            )
            negative_integrals, compute_info_negative, integration_domains_negative, negative_occ_probs = compute_integrals(
                y_cont_neg,
                nprob=True,
                res=self.config.loss.integral_resolution,
                data=data,
                time_horizon=time_horizon
            )
            data.integration_domains_positive = integration_domains_positive
            data.positive_integrals = positive_integrals
            data.integration_domains_negative = integration_domains_negative
            data.negative_integrals = negative_integrals

        if self.config.path_conditioning:
            positive_integrals_path, compute_info_positive_path, integration_domains_positive_path, positive_occ_probs_path = compute_integrals(
                y_cont_path,
                res=self.config.loss.integral_resolution,
                data=data,
                path_conditioning=True,
                time_horizon=time_horizon
            )
            negative_integrals_path, compute_info_negative_path, integration_domains_negative_path, negative_occ_probs_path = compute_integrals(
                y_cont_neg_path,
                nprob=True,
                res=self.config.loss.integral_resolution,
                data=data,
                path_conditioning=True,
                time_horizon=time_horizon
            )
            data.integration_domains_positive_path = integration_domains_positive_path
            data.positive_integrals_path = positive_integrals_path
            data.integration_domains_negative_path = integration_domains_negative_path
            data.negative_integrals_path = negative_integrals_path


        # positive_samples = y_disc.sum(axis=(1, 2))
        # negative_samples = resolution*time_horizon - positive_samples
        # pred_pos: Tensor = y_disc*occ_probs
        # pred_neg: Tensor = (1 - y_disc)*(1 - occ_probs)
        # sum_pred_pos = pred_pos.sum(axis=(1, 2))
        # sum_pred_neg = pred_neg.sum(axis=(1, 2))
        # avg_pred_pos = sum_pred_pos / torch.clamp(positive_samples, min=1)
        # avg_pred_neg = sum_pred_neg / torch.clamp(negative_samples, min=1)
        #log_pred_pos: Tensor = y_disc*torch.log(occ_probs + 1e-7)
        #log_pred_neg: Tensor = (1 - y_disc)*torch.log((1 - occ_probs + 1e-7))
        # sum_log_pred_pos = log_pred_pos.sum(axis=(1, 2))
        # sum_log_pred_neg = log_pred_neg.sum(axis=(1, 2))
        # avg_log_pred_pos = sum_log_pred_pos / torch.clamp(positive_samples, min=1)
        # avg_log_pred_neg = sum_log_pred_neg / torch.clamp(negative_samples, min=1)
        # weighted_log_pred = 0.5*avg_log_pred_pos + 0.5*avg_log_pred_neg
        # # mean_sample_occupancies = y_disc.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1).repeat(1, y_disc.shape[1], y_disc.shape[2])
        # # sample_occupancy_abs_weights = torch.exp(5*mean_sample_occupancies)
        # # sample_occupancy_weights = sample_occupancy_abs_weights/sample_occupancy_abs_weights.mean()
        # #neg_log_probs = -(correct_log_probs*time_discount_vec*sample_occupancy_weights)
        # neg_log_probs = -(weighted_log_pred*time_discount_vec)
        # mean_neg_log_prob = (neg_log_probs).mean()
        # theta_raw = compute_info['theta_raw']
        # theta_norms = torch.norm(theta_raw, 2, dim=(0, 1))
        # mean_theta_norm = theta_norms.mean()

        # correct_probs = y_disc*occ_probs + (1 - y_disc)*(1 - occ_probs)
        # correct_log_probs = torch.log(torch.clamp(correct_probs, min=1e-7))
        
        #mean_sample_occupancies = y_disc.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1).repeat(1, y_disc.shape[1], y_disc.shape[2])
        #sample_occupancy_abs_weights = torch.exp(5*mean_sample_occupancies)
        #sample_occupancy_weights = sample_occupancy_abs_weights/sample_occupancy_abs_weights.mean()
        
        # w = compute_info_positive['w']
        # beta = compute_info_positive['beta']
        # alpha = compute_info_positive['alpha']
        # delta = compute_info_positive['delta']
        # entropy_loss = 0.1*beta.norm(1, dim=-1).mean()
        # entropy_beta = (w*torch.log(beta + 1)).sum(dim=-1)
        # entropy_alpha = (w*torch.log(beta + 1)).sum(dim=-1)
        # entropy = entropy_beta.mean() + entropy_alpha.mean()
        # entropy_loss = 5*entropy

        if self.config.variational:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
            kld_loss = 0.0

        # w_weights = torch.arange(self.config.num_ghosts, device=w.device) / ((self.config.num_ghosts - 1)/2)
        # w_weighted = w*w_weights
        # w_l1 = w_weighted.norm(1, dim=-1).mean()
        # w_l2 = w_weighted.norm(2, dim=-1).mean()
        #w_sum = w.sum(dim=-1, keepdim=True)
        #delta_loss = 200*(w*delta.norm(2, dim=-1)/w_sum).mean()

        self.time_discount_vec = self.time_discount_vec.to(device)

        if self.config.loss.density_weighting:
            mean_occupancy_timesteps = y_cont[:, :, :, 2].sum(dim=2)
            relevance_weighting = torch.clamp(self.time_discount_vec*mean_occupancy_timesteps, self.config.loss.min_relevance_weight)
            nll_p = -(positive_integrals*relevance_weighting).mean()
            nll_n = -(negative_integrals*relevance_weighting).mean()
        else:
            if not self.config.path_conditioning or self.config.loss.path_auxillary_loss:
                nll_p = -(positive_integrals*self.time_discount_vec).mean()
                nll_n = -(negative_integrals*self.time_discount_vec).mean()
            if self.config.path_conditioning:
                nll_p_path = -(positive_integrals_path*self.time_discount_vec.to(positive_integrals_path.device)).mean()
                nll_n_path = -(negative_integrals_path*self.time_discount_vec).mean()

        kld_term = self.config.loss.kld_weight * kld_loss
        #correct_log_integrals = 2.0*positive_integrals + 0.5*negative_integrals
        #mean_neg_log_prob = (neg_log_probs).mean()
        #loss = -1.2*nll_p -1*nll_n + 0.1*w_l1 + 0.1*entropy_loss #+ delta_loss#+ w_l1 + w_l2 + entropy_loss + delta_loss

        if not self.config.path_conditioning or self.config.loss.path_auxillary_loss:
            l1_loss = z_lanelet.abs().mean()
            l2_loss = (z_lanelet**2).mean()

            loss = self.config.loss.positive_loss_weight*nll_p + (1 - self.config.loss.positive_loss_weight) * nll_n
        else:
            loss = 0.0
        if self.config.path_conditioning:
            # w = compute_info_positive_path['w']
            # beta = compute_info_positive_path['beta']
            # sparsity_loss = self.config.sparsity_penalty_coef*w.mean()
            # beta_loss = self.config.beta_penalty_coef*(w*torch.log(beta + 1)).mean()

            l1_loss = z_ego_route.abs().mean()
            l2_loss = (z_ego_route**2).mean()

            loss_nll_path = self.config.loss.positive_loss_weight*nll_p_path + (1 - self.config.loss.positive_loss_weight) * nll_n_path # + sparsity_loss + beta_loss
            loss += loss_nll_path
    
        reg_term = self.config.loss.l2_penalty_coef * l2_loss + self.config.loss.l1_penalty_coef * l1_loss
        
        loss += reg_term + kld_term #+ 0.3*w_l1 #+ delta_loss#+ w_l1 + w_l2 + entropy_loss + delta_loss
        info = dict(
            r=reg_term,
            # pl=nll_p.mean(),
            # nl=nll_n.mean(),
            #ent=entropy,
            #g=sum_gradient,
            #w=sum_weights,
            #l2=l2_loss,
            #wl1=w_l1,
            #wl2=w_l2,
            #dta=delta_loss,
            #mstp=occ_probs.std(axis=2).mean(),
            #sstp=occ_probs.std(axis=2).std(),
            #avp=occ_probs.mean(axis=2).mean(),
            #occ=y_disc.mean(),
            #mtn=mean_theta_norm,
            #max_weights=torch.amax(compute_info_positive['theta_raw'].abs(), axis=(0, 1)),
            #avg_weights=torch.mean(compute_info_positive['theta_raw'], axis=(0, 1)),
            #std_weights=torch.std(compute_info_positive['theta_raw'], axis=(0, 1))
        )
        if self.config.variational:
            info['kld'] = kld_loss
        if not self.config.path_conditioning or self.config.loss.path_auxillary_loss:
            info.update(dict(
                pl=nll_p.mean(),
                nl=nll_n.mean(),
                mp=positive_occ_probs.mean(),
                mn=negative_occ_probs.mean(),
            ))
        if self.config.path_conditioning:
            info.update(dict(
                plp=nll_p_path.mean(),
                nlp=nll_n_path.mean(),
                mp=positive_occ_probs_path.mean(),
                mn=negative_occ_probs_path.mean(),
                # theta_mean=compute_info_positive_path['theta_raw_means'].detach(),
                # theta_std=compute_info_positive_path['theta_raw_stds'].detach(),
                #sl=sparsity_loss.item(),
                #bl=beta_loss.item()
            ))

        assert loss.isfinite()

        return loss, info
