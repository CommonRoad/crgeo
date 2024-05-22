from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch import nn

from commonroad_geometric.dataset.commonroad_data import CommonRoadData


@dataclass
class BaseOccupancyEncoderConfig(ABC):
    ...


class BaseOccupancyEncoder(nn.Module, ABC):
    def __init__(
        self,
        output_size: int,
        offset_conditioning: bool,
        velocity_conditioning: bool
    ):
        self.output_size = output_size
        self.offset_conditioning = offset_conditioning
        self.velocity_conditioning = velocity_conditioning
        super(BaseOccupancyEncoder, self).__init__()

    def reset_parameters(self) -> None:
        """Initializes model weights.
        """
        for n, p in self.named_parameters():
            if n.endswith('bias'):
                nn.init.zeros_(p)
            elif p.ndim == 1:
                nn.init.normal_(p)
            else:
                nn.init.xavier_normal_(p, gain=0.5)

    @abstractmethod
    def build(
        self,
        data: CommonRoadData,
        trial = None
    ) -> None:
        ...
