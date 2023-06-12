from abc import ABC, abstractmethod
from typing import Dict, Optional

import gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor, nn

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData

NORMALIZE_EPS = 1e-8


class BaseGeometricFeatureExtractor(ABC, BaseFeaturesExtractor, AutoReprMixin, StringResolverMixin):
    """Base class for graph-based feature extractors

    We need the feature extractor to process the graph data before feeding into the policy network.

    See the Stable Baselines documentation for a comprehensive description of what custom feature extractors offer:
     https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(
        self,
        observation_space: gym.Space,
        normalize_output: bool = False
    ):
        super(BaseGeometricFeatureExtractor, self).__init__(observation_space, self.output_dim)
        self._build(observation_space)
        self.reset_parameters()
        self._normalizer: Optional[RunningMeanStd] = None
        if normalize_output:
            raise NotImplementedError("How to deal with export & import?")
            # self._normalizer = RunningMeanStd(shape=self.output_dim)

    @abstractmethod
    def _build(self, observation_space: gym.Space) -> None: 
        """
        Build the feature extractor. Initialize PyTorch components here to avoid
        "AttributeError: cannot assign module before Module.__init__() call".

        Args:
            observation_space (gym.Space): Observation space.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Should return the size of the extracted feature tensor.
        """

    def reset_parameters(self) -> None:
        """Initializes model weights.
        """
        for n, p in self.named_parameters():
            if n.endswith('bias'):
                nn.init.zeros_(p)
            elif p.ndim == 1:
                nn.init.normal_(p)
            else:
                # TODO fix
                #nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_normal_(p, gain=0.5)

    def forward(self, obs: Dict[str, Tensor]) -> Tensor: # type: ignore
        # Converting the flattened observation dict back into a PyTorch Geometric data instance.
        # TODO: Use simplfied feature indexing when reconstruct method returns CommonRoadData
        data = CommonRoadData.reconstruct(obs)
        out = self._forward(data=data)
        if self._normalizer is not None:
            return self._normalize(out)
        return out

    @abstractmethod
    def _forward(self, data: CommonRoadData) -> Tensor:
        ...

    def _normalize(self, z: Tensor) -> Tensor:
        assert self._normalizer is not None
        mu, std = self._normalizer.mean, self._normalizer.var ** 0.5
        z_norm = (z -mu) / std
        return z_norm
