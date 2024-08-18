from typing import Dict, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data

from commonroad_geometric.common.config import Config
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from projects.geometric_models.drivable_area.models.decoder.occupancy_decoder import OccupancyDecoder
from projects.geometric_models.drivable_area.models.decoder.road_coverage_prediction import LaneletNetworkGNN


class RoadCoveragePredictionModel(BaseGeometric):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def _build(
        self,
        batch,
        trial
    ) -> None:
        self.encoder = LaneletNetworkGNN(cfg=self.cfg)
        self.decoder = OccupancyDecoder(cfg=self.cfg.drivable_area_decoder)

    def forward(
        self,
        batch: Union[Data, Batch],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.encoder(batch)
        return self.decoder((batch, x))

    def compute_loss(self, data: Batch, output: Tuple[Tensor, Optional[Tensor]]) -> Tuple[Tensor, Dict[str, Tensor]]:
        predicted_road_coverage, sample_ind = output
        if sample_ind is not None:
            n = sample_ind.size(0)
            true_road_coverage = data.vehicle.road_coverage[sample_ind].view(n, -1)
        else:
            n = data.vehicle.road_coverage.size(0)
            true_road_coverage = data.vehicle.road_coverage.view(n, -1)

        loss = F.binary_cross_entropy(input=predicted_road_coverage, target=true_road_coverage)
        info = {}
        return loss, info
