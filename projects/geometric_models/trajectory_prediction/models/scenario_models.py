from __future__ import annotations

import random
from typing import Dict, List, Tuple, Union, Optional

from torch import Tensor
from torch.nn import Tanh, Identity

from commonroad_geometric.common.config import Config
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin
from commonroad_geometric.rendering.plugins.render_traffic_graph_plugin import RenderTrafficGraphPlugin
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from projects.geometric_models.drivable_area.models.decoder.drivable_area_decoder import DrivableAreaDecoder
from projects.geometric_models.drivable_area.models.encoder.scenario_encoder import ScenarioEncoderModel
from projects.geometric_models.drivable_area.utils.visualization.render_plugins import RenderDrivableAreaPlugin
from torch_geometric.nn import BatchNorm

class ScenarioDrivableAreaModel(BaseGeometric):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def forward(
        self,
        data: Union[CommonRoadData, CommonRoadDataTemporal],
    ) -> Tuple[int, Tensor]:
        index = -1
        if isinstance(data, CommonRoadDataTemporal):
            # select a random time step to compute drivable area for
            index = 0 if self.training else random.choice(range(data.num_graphs))
            cr_data = data.get_example(index)
        else:
            cr_data = data
        x_vehicle = self.encoder(cr_data)

        if self.cfg.road_coverage.use_sampling_weights and self.training:
            if isinstance(data, CommonRoadDataTemporal):
                sampling_weights = data.sampling_weights[data.v.batch == index]
            else:
                sampling_weights = data.sampling_weights
        else:
            sampling_weights = None
        prediction = self.decoder(cr_data, x_vehicle, sampling_weights=sampling_weights)
        return index, x_vehicle, prediction

    def compute_loss(
        self,
        prediction: Tuple[int, Tensor],
        data: Union[CommonRoadData, CommonRoadDataTemporal],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        time_step, x_vehicle, prediction = prediction
        if isinstance(data, CommonRoadDataTemporal):
            data = data.get_example(time_step)

        primary_loss = self.decoder.compute_loss(data, prediction)
        binary_loss = self.decoder.compute_binary_loss(data, prediction)

        info = dict(
            binary_loss=binary_loss
        )

        return primary_loss, info

    def _build(
        self,
        batch: CommonRoadData,
        trial = None
    ) -> None:
        self.encoder = ScenarioEncoderModel(cfg=self.cfg)
        self.decoder = DrivableAreaDecoder(cfg=self.cfg.drivable_area_decoder)

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRendererPlugin]]:
        return [
            RenderLaneletNetworkPlugin(),
            RenderObstaclesPlugin(),
            RenderDrivableAreaPlugin(alpha=0.5),
            RenderTrafficGraphPlugin(),
        ]