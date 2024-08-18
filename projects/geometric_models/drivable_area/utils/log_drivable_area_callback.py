from typing import List

import torch
import wandb
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, EarlyStoppingCallbacksParams, StepCallbackParams
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
from projects.geometric_models.drivable_area.utils.visualization.plotting import create_drivable_area_prediction_image


class LogDrivableAreaWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self, 
        wandb_service: WandbService, 
        target_attribute: str = "occupancy"
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.target_attribute = target_attribute

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[Tensor] = []
        try:
            time_step, prediction = params.output[1], params.output[2]
        except IndexError:
            return  # TODO, trajectory prediction model
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        try:
            prediction = prediction[self.target_attribute]
        except KeyError:
            return
        target = params.batch.vehicle[self.target_attribute]
        assert prediction.size(0) == params.batch.v[self.target_attribute].size(0)

        prediction = prediction.view(prediction.size(0), prediction_size, prediction_size)
        prediction_rgb = create_drivable_area_prediction_image(prediction).numpy()
        # torch.randint(0, prediction.size(0), size=(12,)):
        for idx in range(min(30, target.size(0))):
            prediction_img = prediction_rgb[idx].astype(int)
            drivable_area_img = (target[idx] * 255).type(torch.uint8).cpu().numpy().astype(int)
            images += [
                # wandb.data_types.Image(batch.road_coverage[idx], mode="L", caption="actual"),
                wandb.data_types.Image(drivable_area_img, mode="L", caption="actual"),
                wandb.data_types.Image(prediction_img, mode="RGB", caption="predicted"),
            ]
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(figsize=(6, 3.2), ncols=2)
            # axes[0].set_title('true')
            # axes[0].imshow(drivable_area_img)
            # axes[1].set_title('predicted')
            # axes[1].imshow(prediction_img)
            # #ax.set_aspect('equal')
            # plt.show()

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )

        return


class LogOccupancyFlowWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self, 
        wandb_service: WandbService,
        occupancy_attribute: str = "occupancy",
        target_attribute: str = "occupancy_flow"
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.occupancy_attribute = occupancy_attribute
        self.target_attribute = target_attribute

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[Tensor] = []
        try:
            time_step, prediction = params.output[1], params.output[2]
        except IndexError:
            return  # TODO, trajectory prediction model
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        try:
            prediction = prediction[self.target_attribute]
        except KeyError:
            return
        
        target = params.batch.vehicle[self.target_attribute]
        assert prediction.size(0) == target.size(0)

        prediction = prediction.view(prediction.size(0), prediction_size, prediction_size, 2)
        # torch.randint(0, prediction.size(0), size=(12,)):
        for idx in range(min(30, target.size(0))):

            occupancy_mask = (params.batch.vehicle[self.occupancy_attribute][idx, ...] == 0).int()

            true_relative_velocity_0 = target[idx, :, :, 0]
            true_relative_velocity_1 = target[idx, :, :, 1]
            true_relative_velocity_rb_image = torch.clip(0.5 + torch.stack([true_relative_velocity_0, true_relative_velocity_1], dim=-1)/10, 0.0, 1.0)
            target_img = torch.cat([true_relative_velocity_rb_image, torch.ones_like(occupancy_mask[:, :, None])], dim=-1)
            target_img_np = (target_img.detach().cpu().numpy()*255).astype(int)

            pred_relative_velocity_0 = prediction[idx, :, :, 0]
            pred_relative_velocity_1 = prediction[idx, :, :, 1]
            pred_relative_velocity_rb_image = torch.clip(0.5 + torch.stack([pred_relative_velocity_0, pred_relative_velocity_1], dim=-1)/10, 0.0, 1.0)
            pred_img = torch.cat([pred_relative_velocity_rb_image, torch.ones_like(occupancy_mask[:, :, None])], dim=-1)
            pred_img_np = (pred_img.detach().cpu().numpy()*255).astype(int)

            images += [
                # wandb.data_types.Image(batch.road_coverage[idx], mode="L", caption="actual"),
                wandb.data_types.Image(target_img_np, mode="RGB", caption="actual"),
                wandb.data_types.Image(pred_img_np, mode="RGB", caption="predicted"),
            ]
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(figsize=(6, 3.2), ncols=2)
            # axes[0].set_title('true')
            # axes[0].imshow(target_img_np)
            # axes[1].set_title('predicted')
            # axes[1].imshow(pred_img_np)
            # #ax.set_aspect('equal')
            # plt.show()

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )

        return


class LogDrivableAreaTemporalWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self,
        wandb_service: WandbService,
        target_attribute: str = "occupancy",
        max_samples: int = 4,
        max_timesteps: int = 5
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.target_attribute = target_attribute
        self.max_samples = max_samples
        self.max_timesteps = max_timesteps
        self._plt_counter = 0

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[wandb.Image] = []
        z, predictions, target_sequences = params.output
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        try:
            prediction = predictions[self.target_attribute]
            target = target_sequences[self.target_attribute]
        except KeyError:
            return

        assert target.size() == prediction.size()

        batch_size, sequence_length, height, width = target.size()

        # Select a subset of samples and timesteps
        samples = min(self.max_samples, batch_size)
        timesteps = min(self.max_timesteps, sequence_length)

        for sample_idx in range(samples):
            self.fig = plt.figure(figsize=(4*timesteps, 8))
            gs = GridSpec(2, timesteps, height_ratios=[1, 1], wspace=0.05, hspace=0.2)

            self.fig.suptitle(f"Vehicle {sample_idx} {self.target_attribute.capitalize()} (Iteration {self._plt_counter})", fontsize=16)

            for t in range(timesteps):
                # Plot ground truth
                ax_gt = self.fig.add_subplot(gs[0, t])
                ax_gt.imshow(target[sample_idx, t].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
                ax_gt.axis('off')

                # Plot prediction
                ax_pred = self.fig.add_subplot(gs[1, t])
                ax_pred.imshow(prediction[sample_idx, t].detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
                ax_pred.axis('off')

                # Add time-step label underneath
                self.fig.text((t + 0.5) / timesteps, 0.01, f"T{t}", ha='center', va='center', fontsize=10)

            # Add row labels
            self.fig.text(0.02, 0.75, 'Ground Truth', ha='left', va='center', fontsize=14, rotation=90)
            self.fig.text(0.02, 0.25, 'Predicted', ha='left', va='center', fontsize=14, rotation=90)

            self.fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])  # Adjust the rect to accommodate labels

            # Log the figure to wandb
            images.append(wandb.Image(self.fig))

            plt.close(self.fig)

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )
        self._plt_counter += 1


class LogOccupancyFlowTemporalWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(
        self, 
        wandb_service: WandbService,
        occupancy_attribute: str = "occupancy",
        target_attribute: str = "occupancy_flow",
        max_samples: int = 4,
        max_timesteps: int = 5
    ):
        super().__init__()
        self.wandb_service = wandb_service
        self.occupancy_attribute = occupancy_attribute
        self.target_attribute = target_attribute
        self.max_samples = max_samples
        self.max_timesteps = max_timesteps
        self._plt_counter = 0

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[wandb.Image] = []
        z, predictions, target_sequences = params.output

        try:
            prediction = predictions[self.target_attribute]
            target = target_sequences[self.target_attribute]
            occupancy = target_sequences[self.occupancy_attribute]
        except KeyError:
            return

        assert target.size() == prediction.size()

        batch_size, sequence_length, height, width, channels = target.size()
        assert channels == 2, "Expected 2 channels for flow (x and y components)"

        # Select a subset of samples and timesteps
        samples = min(self.max_samples, batch_size)
        timesteps = min(self.max_timesteps, sequence_length)

        for sample_idx in range(samples):
            # Increase the figure size
            fig = plt.figure(figsize=(6*timesteps, 18))
            gs = GridSpec(3, timesteps, height_ratios=[1, 1, 1], wspace=0.1, hspace=0.3)

            fig.suptitle(f"Vehicle {sample_idx} Occupancy Flow (Iteration {self._plt_counter})", fontsize=20)

            for t in range(timesteps):
                # Plot ground truth
                ax_gt = fig.add_subplot(gs[0, t])
                self._plot_flow(ax_gt, target[sample_idx, t], occupancy[sample_idx, t])
                ax_gt.set_title(f"T{t}", fontsize=16)

                # Plot prediction
                ax_pred = fig.add_subplot(gs[1, t])
                self._plot_flow(ax_pred, prediction[sample_idx, t], occupancy[sample_idx, t])

                # Plot difference
                ax_diff = fig.add_subplot(gs[2, t])
                self._plot_flow_difference(ax_diff, target[sample_idx, t], prediction[sample_idx, t], occupancy[sample_idx, t])

            # Add row labels with larger font size
            fig.text(0.02, 0.85, 'Ground Truth', ha='left', va='center', fontsize=18, rotation=90)
            fig.text(0.02, 0.5, 'Predicted', ha='left', va='center', fontsize=18, rotation=90)
            fig.text(0.02, 0.15, 'Difference', ha='left', va='center', fontsize=18, rotation=90)

            fig.tight_layout(rect=[0.04, 0.03, 0.98, 0.95])  # Adjust the rect to accommodate labels

            # Log the figure to wandb
            images.append(wandb.Image(fig))

            # plt.show()

            plt.close(fig)

        self.wandb_service.log(
            {'img_' + self.target_attribute: images}, step=params.ctx.step
        )
        self._plt_counter += 1

    def _plot_flow(self, ax, flow, occupancy):
        # Normalize flow vectors
        magnitude = torch.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        max_magnitude = torch.max(magnitude)
        normalized_flow = flow / (max_magnitude + 1e-8)  # Add small epsilon to avoid division by zero

        # Create a mask for occupied areas
        mask = occupancy.squeeze() > 0.5

        normalized_flow = normalized_flow.cpu().numpy()
        mask = mask.cpu().numpy()

        # Plot flow vectors with longer arrows
        ax.quiver(normalized_flow[..., 0] * mask, normalized_flow[..., 1] * mask, 
                  scale=7, scale_units='inches', color='blue', alpha=0.7, width=0.003)
        
        # Plot occupancy
        ax.imshow(occupancy.squeeze().cpu().numpy(), cmap='Greys', alpha=0.3)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_flow_difference(self, ax, target_flow, pred_flow, occupancy):
        # Calculate difference
        diff_flow = target_flow - pred_flow

        # Normalize difference vectors
        magnitude = torch.sqrt(diff_flow[..., 0]**2 + diff_flow[..., 1]**2)
        max_magnitude = torch.max(magnitude)
        normalized_diff = diff_flow / (max_magnitude + 1e-8)  # Add small epsilon to avoid division by zero

        # Create a mask for occupied areas
        mask = occupancy.squeeze() > 0.5

        mask = mask.detach().cpu().numpy()
        normalized_diff = normalized_diff.detach().cpu().numpy()

        # Plot difference vectors with longer arrows
        ax.quiver(normalized_diff[..., 0] * mask, normalized_diff[..., 1] * mask, 
                  scale=7, scale_units='inches', color='red', alpha=0.7, width=0.003)
        
        # Plot occupancy
        ax.imshow(occupancy.squeeze().cpu().numpy(), cmap='Greys', alpha=0.3)
        ax.set_aspect('equal')
        ax.axis('off')