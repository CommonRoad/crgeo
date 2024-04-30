from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from torch import Tensor, nn

from commonroad_geometric.dataset.commonroad_data import CommonRoadData


@dataclass
class BaseOccupancyDecoderConfig(ABC):
    ...


class BaseOccupancyDecoder(nn.Module, ABC):

    def __init__(self):
        super(BaseOccupancyDecoder, self).__init__()

    @abstractmethod
    def build(
        self,
        data: CommonRoadData,
        trial = None
    ) -> None:
        ...

    def reset_parameters(self) -> None:
        """Initializes model weights.
        """
        for n, p in self.named_parameters():
            if n.endswith('bias'):
                # nn.init.normal_(p, std=0.3)
                nn.init.zeros_(p)
            elif p.ndim == 1:
                nn.init.normal_(p, std=1.0)
            else:
                nn.init.xavier_normal_(p, gain=0.5)

    @abstractmethod
    def forward(
        self,
        lanelet_length: Union[float, Tensor],
        domain: Union[int, Tensor],
        z: Tensor,
        dt: float,
        time_horizon: int
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes probabilistic future occupancies over lanelets.

        Args:
            lanelet_length (Tensor):
                Length tensor for lanelets.
            domain (Union[int, Tensor]):
                Spatial computation domain. If an integer is specified, a
                discrete grid with this resolution will be created automatically.
            z (Tensor):
                Encoding as computed by forward method.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                Tuple containing occupancy probability tensor and info dictionary.
                The tensor dimensions are [H, B, T, S], where
                    H is number of per-batch unique computation domains,
                    B is the batch size,
                    T is the discrete time horizon,
                    S is the spatial dimension
        """
