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
from projects.geometric_models.drivable_area.models.decoder.temporal_occupancy_flow_decoder import TemporalOccupancyFlowDecoder
from projects.geometric_models.drivable_area.models.decoder.temporal_occupancy_decoder import TemporalOccupancyDecoder
from projects.geometric_models.drivable_area.models.encoder.scenario_encoder import ScenarioEncoderModel
from projects.geometric_models.drivable_area.utils.visualization.render_plugins import RenderDrivableAreaPlugin
from projects.geometric_models.drivable_area.utils.vectorized_occupancy_post_processor import save_compute_occupancy_vectorized

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import numpy as np

import torch


class MLPContextualizer(nn.Module):
    def __init__(self, latent_dim, context_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + context_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, x, context):
        combined = torch.cat([x, context], dim=-1)
        return self.mlp(combined)


class AttentionContextualizer(nn.Module):
    def __init__(self, latent_dim, context_dim):
        super(AttentionContextualizer, self).__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        # Define the layers for computing attention scores
        self.query_layer = nn.Linear(latent_dim, latent_dim)
        self.key_layer = nn.Linear(context_dim, latent_dim)
        self.value_layer = nn.Linear(context_dim, latent_dim)
        self.output_layer = nn.Linear(latent_dim + latent_dim, latent_dim)
    
    def forward(self, x, context):
        # Compute query, key, and value
        query = self.query_layer(x)
        key = self.key_layer(context)
        value = self.value_layer(context)
        
        # Compute attention scores
        attention_scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)).squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute the context vector as a weighted sum of values
        context_vector = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)
        
        # Concatenate the original latent representation with the context vector
        combined = torch.cat((x, context_vector), dim=1)
        
        # Apply a linear transformation to the combined vector
        output = self.output_layer(combined)
        
        return output

class FiLMContextualizer(nn.Module):
    def __init__(self, latent_dim, context_dim):
        super(FiLMContextualizer, self).__init__()
        self.gamma = nn.Linear(context_dim, latent_dim)
        self.beta = nn.Linear(context_dim, latent_dim)
    
    def forward(self, x, context):
        gamma = self.gamma(context)
        beta = self.beta(context)
        return gamma * x + beta

def extract_unique_ids_and_presence(ids, batches):
    
    # Get unique ids and the inverse indices
    unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
    
    # Get unique batch indices
    unique_batches = torch.unique(batches)
    
    # Initialize presence matrix with zeros
    presence = torch.zeros(len(unique_batches), len(unique_ids), dtype=torch.bool)
    
    # Update presence matrix
    presence[batches, inverse_indices] = True
    
    return unique_ids, presence

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

class ScenarioTemporalDrivableAreaModel(BaseGeometric):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def prepare_data(self, data: CommonRoadDataTemporal) -> Tuple[CommonRoadDataTemporal, CommonRoadDataTemporal, CommonRoadDataTemporal, torch.Tensor]:
        """
        Prepare the data by splitting it into observation, cutoff, and prediction parts.
        
        Args:
            data (CommonRoadDataTemporal): The input temporal data.
        
        Returns:
            Tuple containing observation data, cutoff data, prediction data, and always present vehicle mask.
        """
        cutoff_index = data.num_graphs // 2
        data_obs = data.get_time_window(slice(0, cutoff_index + 1))
        data_cutoff = data.get_example(cutoff_index)
        data_pred = data.get_time_window(slice(cutoff_index, data.num_graphs + 1))

        ids_pred = data_pred.v.id.squeeze()
        ids_cutoff = data_cutoff.v.id.squeeze()

        batches = data_pred.v.batch
        unique_ids, presence_tensor = extract_unique_ids_and_presence(ids_pred, batches)
        always_present_mask = torch.all(presence_tensor, dim=0)
        always_present_ids = unique_ids[always_present_mask]

        return data_obs, data_cutoff, data_pred, always_present_ids

    def encode_and_contextualize(self, data_obs: CommonRoadDataTemporal, data_cutoff: CommonRoadDataTemporal, always_present_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode the observation data and contextualize it.
        
        Args:
            data_obs (CommonRoadDataTemporal): The observation data.
            data_cutoff (CommonRoadDataTemporal): The cutoff data.
            always_present_ids (torch.Tensor): Tensor of IDs for vehicles always present.
        
        Returns:
            Contextualized vehicle encodings.
        """
        x_vehicle = self.encoder(data_obs)
        mask_obs = data_obs.v.batch == data_obs.num_graphs - 1
        mask_cutoff = torch.isin(data_cutoff.v.id.squeeze(), always_present_ids)

        velocity_key = "view_velocity_long" if "view_velocity_long" in data_cutoff.v else data_cutoff.v.velocity
        
        x_vehicle_t = x_vehicle[mask_obs][mask_cutoff]
        if "view_velocity_long" in data_cutoff.v:
            view_context = torch.cat([
                (data_cutoff.v.view_velocity_long - 23.24) / 5.83,
                data_cutoff.v.view_steering_angle / 0.17
            ], dim=-1)[mask_cutoff]
        else:
            view_context = torch.cat([
                (data_cutoff.v.velocity[:, (0, )] - 23.24) / 5.83,
                data_cutoff.v.steering_angle  / 0.17
            ], dim=-1)[mask_cutoff]

        return self.contextualizer(x_vehicle_t, view_context)
    
    def create_sequence_tensor(self, data_pred: CommonRoadDataTemporal, always_present_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create tensors containing data sequences for vehicles present throughout the prediction period.
        
        Args:
            data_pred (CommonRoadDataTemporal): The prediction data.
            always_present_ids (torch.Tensor): Tensor of IDs for vehicles present in all time steps.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of tensors containing data sequences for always-present vehicles.
        """
        pred_ids = data_pred.v.id.squeeze()
        pred_batches = data_pred.v.batch
        
        id_to_index = {id.item(): idx for idx, id in enumerate(always_present_ids)}
        vehicle_indices = torch.tensor([id_to_index[id.item()] for id in pred_ids if id.item() in id_to_index],
                                    device=pred_ids.device)
        time_indices = pred_batches[torch.isin(pred_ids, always_present_ids)]
        
        sequence_tensors = {}
        
        for attr in self.decoder_heads.keys():
            data = getattr(data_pred.v, attr)
            
            # Determine the shape of the sequence tensor based on the attribute's shape
            attr_shape = data.shape[1:]  # Exclude the first dimension (number of vehicles)
            sequence_shape = (len(always_present_ids), data_pred.num_graphs) + attr_shape
            
            sequence_tensor = torch.zeros(sequence_shape, device=data.device)
            
            # Use advanced indexing to assign values
            sequence_tensor[vehicle_indices, time_indices, ...] = data[torch.isin(pred_ids, always_present_ids)]
            
            sequence_tensors[attr] = sequence_tensor
        
        return sequence_tensors


    def forward(self, data: CommonRoadDataTemporal) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the ScenarioTemporalDrivableAreaModel.
        
        Args:
            data (CommonRoadDataTemporal): The input temporal data.
        
        Returns:
            Tuple containing encoded vehicle representations, predictions, and occupancy sequence tensor.
        """
        data_obs, data_cutoff, data_pred, always_present_ids = self.prepare_data(data)
        
        z = self.encode_and_contextualize(data_obs, data_cutoff, always_present_ids)
        
        predictions = self.decode(data, z)
        
        target_sequences = self.create_sequence_tensor(data_pred, always_present_ids)

        # Check if debugger is active
        if True and sys.gettrace() is not None:
            if '_plt_counter' not in self.__dict__:
                self.__dict__['_plt_counter'] = 0
                plt.ion()
                self.fig = plt.figure(figsize=(20, 10))
                self.gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
                self.axs = [self.fig.add_subplot(self.gs[i]) for i in range(2)]

            PLT_FREQ = 10
            if self.__dict__['_plt_counter'] % PLT_FREQ == 0:
                # Sample a random vehicle
                vehicle_idx = random.randint(0, len(always_present_ids) - 1)
                
                # Get ground truth and predicted occupancy for the sampled vehicle
                ground_truth = target_sequences['occupancy'][vehicle_idx]
                predicted = predictions['occupancy'][vehicle_idx]

                # Clear previous plot
                self.fig.clear()

                # Plot ground truth and predicted occupancy side by side
                self.plot_occupancy_comparison(ground_truth, predicted, vehicle_idx)

                # Redraw the plot
                plt.draw()
                # Pause to update the plot
                plt.pause(0.1)
            
            self.__dict__['_plt_counter'] += 1

        return z, predictions, target_sequences
    

    def plot_occupancy_comparison(self, ground_truth, predicted, vehicle_idx):
        """
        Plot ground truth and predicted occupancy horizontally for all timesteps.

        Args:
            ground_truth (torch.Tensor): Ground truth occupancy tensor
            predicted (torch.Tensor): Predicted occupancy tensor
            vehicle_idx (int): Index of the vehicle being visualized
        """
        num_timesteps = ground_truth.shape[0]
        
        # Create a grid with 2 rows (ground truth and prediction) and num_timesteps columns
        gs = GridSpec(2, num_timesteps, height_ratios=[1, 1], wspace=0.05, hspace=0.2)
        
        self.fig.suptitle(f"Vehicle {vehicle_idx} Occupancy (Iteration {self.__dict__['_plt_counter']})", fontsize=16)
        
        for t in range(num_timesteps):
            # Plot ground truth
            ax_gt = self.fig.add_subplot(gs[0, t])
            ax_gt.imshow(ground_truth[t].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            ax_gt.axis('off')
            
            # Plot prediction
            ax_pred = self.fig.add_subplot(gs[1, t])
            ax_pred.imshow(predicted[t].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
            ax_pred.axis('off')
            
            # Add time-step label underneath
            self.fig.text((t + 0.5) / num_timesteps, 0.01, f"T{t}", ha='center', va='center', fontsize=10)
        
        # Add row labels
        self.fig.text(0.02, 0.75, 'Ground Truth', ha='left', va='center', fontsize=14, rotation=90)
        self.fig.text(0.02, 0.25, 'Predicted', ha='left', va='center', fontsize=14, rotation=90)

        self.fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])  # Adjust the rect to accommodate labels
    
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
        z, predictions, target_sequences = output

        loss_dict = {}
        for key, decoder in self.decoder_heads.items():
            loss_dict[key] = decoder.compute_loss(target_sequences[key], predictions[key])

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

        if self.cfg.contextualizer.model_cls == "Attention":
            self.contextualizer = AttentionContextualizer(
                latent_dim=self.cfg.graph_features.node_features,
                context_dim=2
            )
        elif self.cfg.contextualizer.model_cls == "FiLM":
            self.contextualizer = FiLMContextualizer(
                latent_dim=self.cfg.graph_features.node_features,
                context_dim=2
            )
        elif self.cfg.contextualizer.model_cls == "MLP":
            self.contextualizer = MLPContextualizer(
                latent_dim=self.cfg.graph_features.node_features,
                context_dim=2
            )
        else:
            raise NotImplementedError(self.cfg.contextualizer.model_cls)

        
        self.decoder_heads = nn.ModuleDict({
            'occupancy': TemporalOccupancyDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="occupancy"),
            # 'occupancy_flow': TemporalOccupancyFlowDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="occupancy_flow", mask_attribute="occupancy"),
            # 'polar_occupancy': TemporalOccupancyDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="polar_occupancy"),
            # 'polar_occupancy_flow': TemporalOccupancyFlowDecoder(cfg=self.cfg.drivable_area_decoder, target_attribute="polar_occupancy_flow", mask_attribute="polar_occupancy"),
        })

    def train_preprocess(
        self,
        data: CommonRoadData
    ) -> CommonRoadData:
        save_compute_occupancy_vectorized(data=data, image_size=self.cfg.drivable_area_decoder.prediction_size)  
        return data

    @classmethod
    def configure_renderer_plugins(cls) -> Optional[List[BaseRenderPlugin]]:
        return [
            RenderLaneletNetworkPlugin(),
            RenderTrafficGraphPlugin(),
            RenderObstaclePlugin(),
            RenderDrivableAreaPlugin(alpha=0.5),
        ]
