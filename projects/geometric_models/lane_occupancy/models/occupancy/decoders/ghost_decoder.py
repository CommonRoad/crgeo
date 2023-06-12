from dataclasses import dataclass
from optuna import Trial
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.geometric.components.decoders.base_decoder import BaseDecoder
from commonroad_geometric.learning.geometric.components.distributions import laplace_cdf, triangular_cdf, uniform_cdf
from commonroad_geometric.learning.geometric.components.mlp import MLP
from commonroad_geometric.learning.geometric.components.decoders import LSTMDecoder, BucketDecoder 
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union
from torch.nn import Tanh
from torch import nn
from torch_geometric.nn import BatchNorm
from torch.nn.parameter import Parameter
import torch
import numpy as np
from torch import Tensor
import math

from projects.geometric_models.lane_occupancy.models.occupancy.decoders.base_occupancy_decoder import BaseOccupancyDecoder, BaseOccupancyDecoderConfig

MIN_BETA = 1e-3

@dataclass
class GhostOccupancyDecoderConfig(BaseOccupancyDecoderConfig):
    activation_cls: Type[nn.Module] = Tanh
    decoder_class: Type[BaseDecoder] = LSTMDecoder
    decoder_hidden_size: int = 256
    decoder_layers: int = 1
    decoding_act: bool = True
    decoding_norm: bool = True
    max_cooling_factor: float = 50.0
    max_decay_factor: float = 20.0
    max_delta: float = 20.0
    max_distr_acc: float = 10.0
    max_distr_speed: float = 50.0
    min_distr_speed: float = 0.01
    max_init_beta: float = 1.0
    min_delta: float = 0.25
    max_mu_offset: float = 1.0
    mlp_layers_decoder: int = 1
    mlp_layers_distr: int = 1
    mlp_hidden_size_decode: int = 256
    mlp_hidden_size_distr: int = 256
    eps: float = 1e3
    norm_method: Literal['layer', 'batch', 'none'] = 'none'
    num_ghosts: int = 12
    prob_distr: Literal['normal', 'logistic', 'gumbel', 'uniform', 'triangular', 'laplace'] = 'normal'
    prob_pooling: Literal['exact', 'max', 'truncadd', 'add'] = 'exact'
    sigmoid_transform: bool = True
    distribution_bias: bool = False
    soft_masking: bool = False


class GhostOccupancyDecoder(BaseOccupancyDecoder):
    N_DISTR_PARAMS = 8

    def __init__(
        self,
        input_size: int,
        offset_conditioning: bool,
        config: Optional[GhostOccupancyDecoderConfig] = None,
        **kwargs: Any
    ):
        config = config or GhostOccupancyDecoderConfig(**kwargs)
        super(GhostOccupancyDecoder, self).__init__()

        self.input_size = input_size
        self.offset_conditioning = offset_conditioning
        self.config = config
        self.decoder: BaseDecoder

        self._log_v_min = math.log(self.config.min_distr_speed)
        self._log_v_max = math.log(self.config.max_distr_speed)
        self._log_delta_min = math.log(self.config.min_delta)
        self._log_delta_max = math.log(self.config.max_delta)

    def reset_config(self) -> None:
        self.config = GhostOccupancyDecoderConfig() # TODO
        
    def build(
        self,
        data: CommonRoadData,
        trial: Optional[Trial] = None
    ) -> None:
        
        batch_size = data.lanelet.batch.max().item() + 1 if hasattr(data.lanelet, "batch") else 1

        if self.config.distribution_bias:
            self.distribution_bias = Parameter(torch.ones((1, self.config.num_ghosts, GhostOccupancyDecoder.N_DISTR_PARAMS)), requires_grad=True)
        else:
            self.distribution_bias = None

        self.linear = nn.Linear(
            in_features=self.input_size,
            out_features=self.config.decoder_hidden_size
        )

        self.z_w_act = self.config.activation_cls() if self.config.decoding_act else nn.Identity()
        self.z_w_norm = BatchNorm(self.config.decoder_hidden_size) if self.config.decoding_norm and batch_size > 1 else nn.Identity()
        self.y_act = nn.Identity() # self.config.activation_cls() if self.config.decoding_act else 

        if self.config.decoder_class is BucketDecoder:
            self.decoder = BucketDecoder(
                input_size=self.config.decoder_hidden_size,
                num_layers=self.config.decoder_layers,
                net_arch=self.config.mlp_layers_decoder*[self.config.mlp_hidden_size_decode],
                #dropout=self.dropout_prob,
                activation_cls=self.config.activation_cls,
                norm_method=self.config.norm_method
            )
        else:
            self.decoder = self.config.decoder_class(
                input_size=self.config.decoder_hidden_size,
                hidden_size=self.config.decoder_hidden_size,
                num_layers=self.config.decoder_layers,
                n=self.config.num_ghosts,
                norm_method=self.config.norm_method,
                #dropout=self.dropout_prob if self.decoder_layers > 1 else 0.0,
                batch_first=True
            )

        self.distr_network = MLP(
            input_size=2*self.config.decoder_hidden_size,
            output_size=GhostOccupancyDecoder.N_DISTR_PARAMS,
            net_arch=self.config.mlp_layers_distr*[self.config.mlp_hidden_size_distr],
            #dropout_prob=0.5,
            activation_cls=Tanh,
            norm_method='none' if batch_size == 1 else self.config.norm_method
            #norm_method=self.norm_method,
        )

    def reset_parameters(self) -> None:
        """Initializes model weights.
        """
        super(GhostOccupancyDecoder, self).reset_parameters()
        if self.config.distribution_bias:
            nn.init.normal_(self.distribution_bias, std=0.1)

    def forward(
        self,
        lanelet_length: Union[float, Tensor],
        domain: Union[int, Tensor],
        z: Tensor,
        dt: float,
        time_horizon: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]: 
        batch_size = z.shape[0]
        theta_raw = self._decode_hidden(z)

        if isinstance(domain, int):
            domain = torch.linspace(
                0, 1, domain, device=theta_raw.device
            )[None, None, None, :, None].repeat(
                1, batch_size, time_horizon, 1, self.config.num_ghosts
            )
            dense_input = False
        else:
            if domain.ndim == 2:
                dense_input = True
            else:
                dense_input = False
                if domain.ndim == 3:
                    domain = domain.unsqueeze(0)

                # BATCH_SIZE x TIME_HORIZON x RESOLUTION x N_DISTR
            domain = domain.unsqueeze(-1)

        distribution_params = self._extract_distribution_params(
            lanelet_length=lanelet_length,
            theta_raw=theta_raw
        )
        mu, beta, w, v, alpha, gamma, tau, delta, a = distribution_params

        mu_t, beta_t, w_t, delta = self._extract_distributions(
            lanelet_length=lanelet_length,
            theta_raw=theta_raw, 
            distribution_params=distribution_params,
            dt=dt,
            time_horizon=time_horizon
        )

        if not dense_input:
            mu_t = mu_t[None, :, :, None, :]
            beta_t = beta_t[None, :, :, None, :]
            w_t = w_t[None, :, :, None, :]
            delta = delta[None, :, :, None, :]
        else:
            mu_t = mu_t[:, :, None, :]
            beta_t = beta_t[:, :, None, :]
            w_t = w_t[:, :, None, :]
            delta = delta[:, :, None, :]

        # TODO: separate convnet for diff lanes vehicles
        # TODO: abs scale for mu, not sigmoid

        eps=self.config.eps

        if self.config.prob_distr == 'gumbel':
            cdf_upper = torch.exp(-torch.exp(torch.clamp(-(domain + delta - mu_t) / beta_t, min=-eps, max=eps)))
            cdf_lower = torch.exp(-torch.exp(torch.clamp(-(domain - delta - mu_t) / beta_t, min=-eps, max=eps)))

        elif self.config.prob_distr == 'logistic':
            cdf_upper = 1 / (1 + torch.exp(torch.clamp(-(domain + delta - mu_t)/beta_t, min=-eps, max=eps)))
            cdf_lower = 1 / (1 + torch.exp(torch.clamp(-(domain - delta - mu_t)/beta_t, min=-eps, max=eps)))

        elif self.config.prob_distr == 'normal':
            cdf_upper = torch.erf(torch.clamp((domain + delta - mu_t) / (beta_t * np.sqrt(2)), min=-eps, max=eps))/2.0
            cdf_lower = torch.erf(torch.clamp((domain - delta - mu_t) / (beta_t * np.sqrt(2)), min=-eps, max=eps))/2.0

        elif self.config.prob_distr == 'uniform':
            cdf_upper = uniform_cdf(
                x=domain + delta,
                m=mu_t,
                s=beta_t
            )
            cdf_lower = uniform_cdf(
                x=domain - delta,
                m=mu_t,
                s=beta_t
            )

        elif self.config.prob_distr == 'triangular':
            cdf_upper = triangular_cdf(
                x=domain + delta,
                m=mu_t,
                s=beta_t
            )
            cdf_lower = triangular_cdf(
                x=domain - delta,
                m=mu_t,
                s=beta_t
            )

        elif self.config.prob_distr == 'laplace':
            cdf_upper = laplace_cdf(
                x=domain + delta,
                m=mu_t,
                s=beta_t
            )
            cdf_lower = laplace_cdf(
                x=domain - delta,
                m=mu_t,
                s=beta_t
            )
            
        else:
            raise NotImplementedError(self.config.prob_distr)

        # N_SLOTS x BATCH_N_LANELETS x TIME_HORIZON x RESOLUTION x N_DISTR
        occ_prob_components = w_t*(cdf_upper - cdf_lower)

        if self.config.prob_pooling == 'exact':
            # Exact = P(one or more)
            occ_probs = (1 - torch.prod(1 - occ_prob_components, dim=-1))

        elif self.config.prob_pooling == 'max':
            # Max pooling
            occ_probs = occ_prob_components.max(dim=-1)[0]

        elif self.config.prob_pooling == 'truncadd':
            # Clipped sum pooling
            occ_probs = torch.clamp(occ_prob_components.sum(dim=-1), max=1.0)

        elif self.config.prob_pooling == 'add':
            # Sum pooling
            occ_probs = occ_prob_components.sum(dim=-1)
        else:
            raise NotImplementedError(self.config.prob_pooling)

        assert occ_probs.min() >= 0.0
        if not self.config.prob_pooling == 'add': 
            assert occ_probs.max() <= 1.0

        info = dict(
            occ_probs=occ_probs,
            occ_prob_components=occ_prob_components,
            theta_raw=theta_raw,
            mu_t=mu_t,
            beta_t=beta_t,
            w_t=w_t,
            delta=delta,
            mu=mu, 
            beta=beta,
            w=w, 
            v=v, 
            alpha=alpha, 
            gamma=gamma, 
            tau=tau,
            a=a,
            theta_raw_means=theta_raw.mean(dim=(0, 1)),
            theta_raw_stds=theta_raw.std(dim=(0, 1))
        )

        # N_SLOTS x BATCH_SIZE x TIME_HORIZON x RESOLUTION
        occ_probs = occ_probs.squeeze(0)
        return occ_probs, info

    def _decode_hidden(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        z_w_raw = self.linear(z)
        z_w = self.z_w_act(self.z_w_norm(z_w_raw)) # watch out for low batch size
        q = self.decoder.forward_n(z_w, n=self.config.num_ghosts)
        y_in_raw = torch.cat([q, z_w.unsqueeze(1).repeat(1, self.config.num_ghosts, 1)], dim=2).flatten(0, 1)
        y_in = self.y_act(y_in_raw)
        y_raw = self.distr_network(y_in).view(
            (batch_size, self.config.num_ghosts, GhostOccupancyDecoder.N_DISTR_PARAMS)
        )
        if self.config.distribution_bias:
            y = y_raw + self.distribution_bias
        else:
            y = y_raw
        return y

    def _extract_distribution_params(
        self,
        lanelet_length: Union[float, Tensor],
        theta_raw: Union[Tensor, np.ndarray]
    ) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        # backward compatibility hacks
        if not hasattr(self, 'offset_conditioning'):
            self.offset_conditioning = False
        if not hasattr(self, 'log_v_min'):
            self._log_v_min = 0 
            self._log_v_max = np.log(self.config.max_distr_speed) 
            self._log_delta_min = np.log(self.config.min_delta)
            self._log_delta_max = np.log(self.config.max_delta)

        #w_multipliers = torch.sigmoid(theta_raw[..., 0])
        if self.config.sigmoid_transform:
            w = torch.sigmoid(theta_raw[..., 0]) #w_multipliers.cumprod(axis=-1)
            if self.offset_conditioning:
                mu = theta_raw[..., 1]
            else:
                mu = -self.config.max_mu_offset + (1.0 + self.config.max_mu_offset) * torch.sigmoid(theta_raw[..., 1]) # torch.clamp(0.5 + theta_raw[..., 1], min=-0.1, max=1.1, )
            beta = torch.sigmoid(theta_raw[..., 2])*self.config.max_init_beta/lanelet_length
            v_log = self._log_v_min + torch.sigmoid(theta_raw[..., 3])*(self._log_v_max - self._log_v_min)
            v = torch.exp(v_log) / lanelet_length
            alpha = torch.sigmoid(theta_raw[..., 4])*self.config.max_cooling_factor/lanelet_length
            gamma = torch.tanh(theta_raw[..., 5])*self.config.max_decay_factor/lanelet_length
            tau = torch.tanh(theta_raw[...,6])
            a = torch.tanh(theta_raw[..., 7])*self.config.max_distr_acc/lanelet_length
        else:
            w = torch.clamp(0.5 + theta_raw[..., 0], 0.0, 1.0) #w_multipliers.cumprod(axis=-1)
            if self.offset_conditioning:
                mu = theta_raw[..., 1]
            else:
                mu = torch.clamp(0.5 + theta_raw[..., 1], 0.0, 1.0) # torch.clamp(0.5 + theta_raw[..., 1], min=-0.1, max=1.1, )
            beta = torch.clamp(0.5 + theta_raw[..., 2], 0.0, 1.0)*self.config.max_init_beta/lanelet_length
            v = torch.clamp(0.5 + theta_raw[..., 3], 0.0, 1.0)*self.config.max_distr_speed/lanelet_length
            alpha = torch.clamp(0.5 + theta_raw[..., 4], 0.0, 1.0)*self.config.max_cooling_factor/lanelet_length
            gamma = torch.clamp(theta_raw[..., 5], -1.0, 1.0)*self.config.max_decay_factor/lanelet_length
            tau = torch.clamp(theta_raw[...,6], -1.0, 1.0)
            a = torch.clamp(theta_raw[..., 7], -1.0, 1.0)*self.config.max_distr_acc/lanelet_length

        if self.config.max_delta != self.config.min_delta:
            # TODO
            delta_log = self._log_delta_min + torch.sigmoid(theta_raw[...,7])*(self._log_delta_max - self._log_delta_min)
            delta = torch.exp(delta_log) / lanelet_length
        else:
            delta = self.config.max_delta/lanelet_length
        return mu, beta, w, v, alpha, gamma, tau, delta, a

    def _extract_distributions(
        self,
        lanelet_length: Union[float, Tensor],
        theta_raw: Tensor,
        dt: float,
        time_horizon: int,
        distribution_params: Optional[Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
        ]]=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = theta_raw.shape[0]

        t_idx = torch.arange(
            time_horizon, device=theta_raw.device
        )[None, :, None].repeat(batch_size, 1, self.config.num_ghosts)
        t = t_idx*dt

        if distribution_params is None:
            mu, beta, w, v, alpha, gamma, tau, delta, a = self._extract_distribution_params(
                lanelet_length=lanelet_length,
                theta_raw=theta_raw
            )
        else:
             mu, beta, w, v, alpha, gamma, tau, delta, a = distribution_params

        if self.config.soft_masking:
            tau_mask = time_horizon*dt*tau[:, None, :]
            R = 6 # TODO: magic
            C = 0.7
            left_mask = torch.sigmoid(R*(t - tau_mask*(1+C) + C))
            right_mask = torch.sigmoid(R*(1 - t + tau_mask*(1+C) + C))
            mask = left_mask*right_mask # TODO: widen default mask
        else:
            mask = 1

        mu_t = mu.unsqueeze(1) + v.unsqueeze(1)*t + 0.5*a.unsqueeze(1)*t**2
        beta_t = torch.clamp(
            beta.unsqueeze(1) + alpha.unsqueeze(1)*t, min=MIN_BETA
        )
        w_t = mask*torch.clamp(w.unsqueeze(1) - gamma.unsqueeze(1)*t, min=0.0, max=1.0)
        # w_t = torch.clamp(w.unsqueeze(1) - gamma.unsqueeze(1)*t, min=0.0, max=1.0) # TODO

        return mu_t, beta_t, w_t, delta.unsqueeze(1)
