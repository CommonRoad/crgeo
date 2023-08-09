from typing import List

import torch
import wandb
from torch import Tensor
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, EarlyStoppingCallbacksParams, StepCallbackParams
from commonroad_geometric.learning.training.wandb_service.wandb_service import WandbService
from projects.geometric_models.drivable_area.utils.visualization.plotting import create_drivable_area_prediction_image



class LogDrivableAreaWandb(BaseCallback[EarlyStoppingCallbacksParams]):
    def __init__(self, wandb_service: WandbService):
        self.wandb_service = wandb_service

    def __call__(self, params: StepCallbackParams) -> None:
        if self.wandb_service.disabled:
            return

        images: List[Tensor] = []
        try:
            time_step, prediction = params.output[-2], params.output[-1]
        except IndexError:
            return # TODO, trajectory prediction model
        if isinstance(prediction, tuple):
            prediction, sample_ind = prediction
        prediction_size = params.ctx.model.cfg.drivable_area_decoder.prediction_size

        assert prediction.size(0) == params.batch.vehicle.drivable_area.size(0)

        prediction = prediction.view(prediction.size(0), prediction_size, prediction_size)
        prediction_rgb = create_drivable_area_prediction_image(prediction).numpy()
        for idx in range(min(30, params.batch.vehicle.drivable_area.size(0))): # torch.randint(0, prediction.size(0), size=(12,)):
            prediction_img = prediction_rgb[idx]
            drivable_area_img = (params.batch.vehicle.drivable_area[idx] * 255).type(torch.uint8).cpu().numpy()
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
            {"img": images}, step=params.ctx.step
        )

        return