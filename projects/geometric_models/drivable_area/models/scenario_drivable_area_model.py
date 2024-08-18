from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from commonroad_geometric.common.torch_utils.geometry import contains_any_rotated_rectangles, relative_angles
from commonroad_geometric.common.config import Config
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.learning.geometric.base_geometric import BaseGeometric
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin, RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from projects.geometric_models.drivable_area.models.decoder.occupancy_flow_decoder import OccupancyFlowDecoder
from projects.geometric_models.drivable_area.models.decoder.occupancy_decoder import OccupancyDecoder
from projects.geometric_models.drivable_area.models.encoder.scenario_encoder import ScenarioEncoderModel
from projects.geometric_models.drivable_area.utils.visualization.render_plugins import RenderDrivableAreaPlugin

import matplotlib.pyplot as plt
import numpy as np

import torch


def reshape_and_visualize_occupancy(occupancy, mask=None):
    """
    Reshape the flat occupancy array and visualize it.
    Assumes the occupancy data should be square.
    """
    # Only proceed if data is flat
    if occupancy.ndim == 1:
        size = int(np.sqrt(occupancy.shape[0]))
        reshaped_occupancy = occupancy.detach().cpu().numpy().reshape(size, size)
    else:
        reshaped_occupancy = occupancy.detach().cpu().numpy()

    # Apply mask if provided
    if mask is not None:
        reshaped_occupancy *= mask

    return np.flip(np.flip(reshaped_occupancy, axis=0), axis=1)

def reshape_and_visualize_occupancy_flow(occupancy_flow, mask=None):
    """
    Reshape the flat occupancy flow array and visualize it.
    """
    occupancy_flow = occupancy_flow.detach().cpu().numpy()
    if occupancy_flow.ndim == 1:
        size = int(np.sqrt(occupancy_flow.shape[0]/2))
        reshaped_flow = occupancy_flow.reshape(size, size, 2)
        vx, vy = np.split(reshaped_flow, 2, axis=2)  # Assuming the flow components are split across the width
    else:
        vx, vy = np.split(occupancy_flow, 2, axis=2)

    # If mask is None, create a ones array of the appropriate shape
    if mask is None:
        mask = np.ones((vx.shape[0], vx.shape[1]), dtype=np.float32)
    else:
        mask = np.clip(mask, 0.2, 1.0)

    # Process like masked_velocity in your original code
    masked_velocity = np.clip(0.5 - np.concatenate([vx, vy], axis=-1) / 10, 0, 1) * mask[..., None]
    velocity_image = np.concatenate([masked_velocity, mask[..., None]], axis=-1)

    return np.flip(np.flip(velocity_image, axis=0), axis=1)

def plot_vehicle_data(i, axs, data, predictions):
    """
    Plots the actual and predicted drivable area, velocity flow, polar drivable area,
    and polar velocity flow for a given vehicle index `i` on the subplot axes `axs`.
    """
    # Extract actual data
    occupancy_flow = data.v.occupancy_flow.detach().cpu().numpy()
    vx, vy = occupancy_flow[i, ..., 0], occupancy_flow[i, ..., 1]
    mask = data.v.occupancy[i, ...].detach().cpu().numpy()

    occupancy_flow_polar = data.v.polar_occupancy_flow.detach().cpu().numpy()
    vx_polar, vy_polar = occupancy_flow_polar[i, ..., 0], occupancy_flow_polar[i, ..., 1]
    mask_polar = data.v.polar_occupancy[i, ...].detach().cpu().numpy()

    # Actual and Predicted Drivable Area
    axs[0, 0].imshow(np.flip(np.flip(mask, axis=0), axis=1), cmap='gray')
    axs[0, 0].set_title(f"Actual Drivable Area - Vehicle {i}")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(reshape_and_visualize_occupancy(predictions['occupancy'][i]))
    axs[0, 1].set_title(f"Predicted Drivable Area - Vehicle {i}")
    axs[0, 1].axis('off')

    # Actual and Predicted Velocity Flow
    masked_velocity = np.clip(0.5 - np.stack([vx, vy], axis=-1) / 10, 0, 1) * mask[..., None]
    image = np.concatenate([masked_velocity, mask[..., None]], axis=-1)
    axs[1, 0].imshow(np.flip(np.flip(image, axis=0), axis=1))
    axs[1, 0].set_title(f"Actual Velocity Flow - Vehicle {i}")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(reshape_and_visualize_occupancy_flow(predictions['occupancy_flow'][i], mask=mask))
    axs[1, 1].set_title(f"Predicted Velocity Flow - Vehicle {i}")
    axs[1, 1].axis('off')

    # Actual and Predicted Polar Drivable Area
    axs[2, 0].imshow(np.flip(np.flip(mask_polar, axis=0), axis=1), cmap='gray')
    axs[2, 0].set_title(f"Actual Polar Drivable Area - Vehicle {i}")
    axs[2, 0].axis('off')
    axs[2, 1].imshow(reshape_and_visualize_occupancy(predictions['polar_occupancy'][i]))
    axs[2, 1].set_title(f"Predicted Polar Drivable Area - Vehicle {i}")
    axs[2, 1].axis('off')

    # Actual and Predicted Polar Velocity Flow
    masked_velocity_polar = np.clip(0.5 - np.stack([vx_polar, vy_polar], axis=-1) / 10, 0, 1) * mask_polar[..., None]
    image_polar = np.concatenate([masked_velocity_polar, mask_polar[..., None]], axis=-1)
    axs[3, 0].imshow(np.flip(np.flip(image_polar, axis=0), axis=1))
    axs[3, 0].set_title(f"Actual Polar Velocity Flow - Vehicle {i}")
    axs[3, 0].axis('off')
    axs[3, 1].imshow(reshape_and_visualize_occupancy_flow(predictions['polar_occupancy_flow'][i], mask=mask_polar))
    axs[3, 1].set_title(f"Predicted Polar Velocity Flow - Vehicle {i}")
    axs[3, 1].axis('off')

def plot_all_vehicles(data, predictions, encodings=None, fig=None, axs=None):
    num_nodes = len(predictions['occupancy'])

    # Determine the number of rows in the subplot: add an extra row if encodings are not None
    if encodings is not None:
        num_rows = 5
    else:
        num_rows = 4

    if fig is None and axs is None:
        fig, axs = plt.subplots(num_rows, 2, figsize=(10, 20))  # 4 types of data, each with actual and predicted

    indices = torch.where(data.v.is_ego_mask)[0].tolist() if "is_ego_mask" in data.v and data.v.is_ego_mask.sum() > 0 else [random.randint(0, num_nodes - 1)]

    for i in indices:
        plot_vehicle_data(i, axs, data, predictions)

    # If encodings are provided, plot them in an additional row
    if encodings is not None:
        for i in indices:
            # Create an axis across both columns for the encodings
            ax = fig.add_subplot(num_rows, 1, num_rows)
            im = ax.imshow(encodings[i].detach().cpu().numpy().reshape(4, -1), aspect="auto", cmap='viridis')  # Show all encodings in one plot
            ax.set_title(f'Encoding - Vehicle {i}')

            # Explicitly remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Turn off the axis (which also removes the border)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


class ScenarioDrivableAreaModel(BaseGeometric):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def forward(
        self,
        data: CommonRoadData,
    ) -> Tuple[int, Tensor]:
        index = -1

        x_vehicle = self.encoder(data)

        sampling_weights = None
        predictions = self.decode(data, x_vehicle, sampling_weights)

        # Check if debugger is active
        import sys
        if False and sys.gettrace() is not None:
            if '_plt_counter' not in self.__dict__:
                self.__dict__['_plt_counter'] = 0
                plt.ion()
                self.fig, self.axs = plt.subplots(5, 2, figsize=(10, 20))  # 4 types of data, each with actual and predicted

            PLT_FREQ = 20
            if self.__dict__['_plt_counter'] % PLT_FREQ == 0:
                plot_all_vehicles(data=data, predictions=predictions, encodings=x_vehicle, fig=self.fig, axs=self.axs)

                # Redraw the plot
                plt.draw()
                # Pause to update the plot
                plt.pause(0.1)
            
            # plot_all_vehicles(data, None)
            self.__dict__['_plt_counter'] += 1

        return index, x_vehicle, predictions
    
    def decode(self, data, x_vehicle, sampling_weights=None) -> dict[str, Tensor]:
        predictions = {}
        for key, decoder in self.decoder_heads.items():
            predictions[key] = decoder(data, x_vehicle, sampling_weights=sampling_weights)
        return predictions

    def compute_loss(
        self,
        output: Tuple[int, Tensor],
        data: Union[CommonRoadData, CommonRoadDataTemporal],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        time_step, x_vehicle, predictions = output
        if isinstance(data, CommonRoadDataTemporal):
            data = data.get_example(time_step)

        loss_dict = {}
        for key, decoder in self.decoder_heads.items():
            loss_dict[key] = decoder.compute_loss(data, predictions[key])

        loss = sum(loss_dict.values())
        loss_dict['mean_occupancy'] = data.v.occupancy.mean()
        loss_dict['mean_prediction'] = predictions['occupancy'].mean()

        return loss, loss_dict

    def _build(
        self,
        batch: CommonRoadData,
        trial=None
    ) -> None:
        self.encoder = ScenarioEncoderModel(cfg=self.cfg)
        self.decoder_heads = nn.ModuleDict({
            'occupancy': OccupancyDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="occupancy"),
            'occupancy_flow': OccupancyFlowDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="occupancy_flow", mask_attribute="occupancy"),
            'polar_occupancy': OccupancyDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="polar_occupancy"),
            'polar_occupancy_flow': OccupancyFlowDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="polar_occupancy_flow", mask_attribute="polar_occupancy"),
        })

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRenderPlugin]]:
        return [
            RenderLaneletNetworkPlugin(),
            RenderTrafficGraphPlugin(),
            RenderObstaclePlugin(),
            RenderDrivableAreaPlugin(alpha=0.5),
        ]