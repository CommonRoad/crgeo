import torch
from gymnasium import Space
from torch import Tensor

from commonroad_geometric.common.config import Config
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.base_geometric_feature_extractor import BaseGeometricFeatureExtractor
#from projects.common.encoder.scenario_encoder import ScenarioEncoderModel # TODO
from projects.geometric_models.drivable_area.models.encoder.scenario_encoder import ScenarioEncoderModel
from projects.geometric_models.drivable_area.models.decoder.occupancy_decoder import OccupancyDecoder
from projects.geometric_models.drivable_area.models.decoder.occupancy_flow_decoder import OccupancyFlowDecoder
from dataclasses import dataclass

@dataclass
class FeatureExtractionDict:
    pred_drivable_area: Tensor
    pred_velocity_flow: Tensor
    recon_loss_drivable_area: Tensor
    recon_loss_velocity_flow: Tensor
    recon_loss: Tensor

class HeteroFeatureExtractor(BaseGeometricFeatureExtractor):

    def __init__(
        self, 
        observation_space: Space,
        encoder_config: Config,
        decoder_config: Config,
        gnn_features: bool,
    ):
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.gnn_features = gnn_features
        super().__init__(observation_space)

    def _build(self, observation_space: Space) -> None:
        if self.gnn_features:
            self.encoder = ScenarioEncoderModel(cfg=self.encoder_config)
            self.drivable_area_decoder = OccupancyDecoder(cfg=self.decoder_config)
            self.velocity_flow_decoder = OccupancyFlowDecoder(cfg=self.decoder_config)

    @property
    def output_dim(self) -> int:
        # output_dim = (6+4*4) + 5 + 3
        output_dim = 6 + 5 + 3
        if self.gnn_features:
            output_dim += self.encoder_config.traffic.vehicle_node_feature_size
        return output_dim

    def _forward(self, data: CommonRoadData) -> Tensor:
        path_observation = data.path_observation
        ego_observation = data.ego_observation
        road_observation = data.road_observation
        goal_observation = data.goal_observation
        x_static = torch.cat([
            # path_observation,
            goal_observation, 
            ego_observation, 
            road_observation
        ], dim=-1)

        if self.gnn_features:
            z = self.encoder.forward(data=data)
            ego_mask = data.vehicle.is_ego_mask.bool().squeeze(-1)
            z_ego = z[ego_mask, :]
            x: Tensor = torch.cat([z_ego, x_static], dim=-1)
        else:
            x = x_static

        if self.encoder.training:
            decoded_drivable_area = self.drivable_area_decoder(data, z_ego)
            decoded_velocity_flow = self.velocity_flow_decoder(data, z_ego)
            reconstruction_loss_drivable_area = self.drivable_area_decoder.compute_loss(data, decoded_drivable_area, only_ego=True)
            reconstruction_loss_velocity_flow = self.velocity_flow_decoder.compute_loss(data, decoded_velocity_flow, only_ego=True)
            reconstruction_loss = reconstruction_loss_drivable_area + reconstruction_loss_velocity_flow
            feature_extraction_dict = FeatureExtractionDict(
                pred_drivable_area=decoded_drivable_area,
                pred_velocity_flow=decoded_velocity_flow,
                recon_loss_drivable_area=reconstruction_loss_drivable_area,
                recon_loss_velocity_flow=reconstruction_loss_velocity_flow,
                recon_loss=reconstruction_loss
            )
        else:
            feature_extraction_dict = None
        assert not x.isnan().any()
        return x, feature_extraction_dict
