from math import sqrt
from typing import Tuple

import numpy as np
import torch
from pyglet import gl
from pyglet.image import ImageData
from torch import Tensor

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D
from projects.geometric_models.drivable_area.utils.visualization.plotting import create_drivable_area_prediction_image


class RenderDrivableAreaPlugin(BaseRenderPlugin):

    def __init__(
        self,
        alpha: float = 1.0,
        render_all: bool = False
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self._alpha = alpha
        self._render_all = render_all
        self._scenario_counter = 0

    @staticmethod
    def _unpack_output(params: RenderParams) -> Tuple[Tensor, Tensor]:
        if 'output' in params.render_kwargs:
            output = params.render_kwargs['output']
            encoding, prediction = output[1], output[2]
        else:
            encoding, prediction = params.data.encoding, params.data.prediction

        if isinstance(prediction, dict):
            prediction = prediction['occupancy']

        return encoding, prediction

    def render(self, viewer: GLViewer2D, params: RenderParams) -> None:
        if params.time_step == params.simulation.initial_time_step:
            self._scenario_counter += 1

        try:
            encoding, prediction = self._unpack_output(params)
        except (IndexError, KeyError, AttributeError):
            return
        
        data = params.data
        if self._render_all:
            for idx in range(data.v.num_nodes):
                self._render_drivable_area(
                    viewer=viewer,
                    position=data.v.pos[idx],
                    orientation=data.v.orientation[idx],
                    drivable_area=prediction[idx],
                )
        else:
            if params.ego_vehicle is not None:
                try:
                    idx = torch.where(data.vehicle.is_ego_mask)[0][-1].item()
                    prediction_to_draw = prediction[0] if prediction.shape[0] == 1 else prediction[idx]
                    state = params.ego_vehicle.state
                except IndexError:
                    prediction_to_draw = None
            else:
                idx = 0
                viewer.focus_obstacle_idx = idx
                prediction_to_draw = prediction[idx]
                state = params.simulation.get_current_obstacle_state(
                    obstacle=params.simulation.current_obstacles[idx]
                )

            if prediction_to_draw is not None:
                # mitigate feature normalization rendering data.v.pos, etc., unusable

                self._render_drivable_area(
                    viewer=viewer,
                    position=state.position,
                    orientation=state.orientation,
                    drivable_area=prediction_to_draw,
                )

    def _render_drivable_area(
        self,
        viewer: GLViewer2D,
        position: Tuple[float, float],
        orientation: float,
        drivable_area: Tensor,
        flip: bool = True
    ) -> None:
        
        if drivable_area.ndim == 3:
            # selecting the first time-step
            drivable_area = drivable_area[0]

        elif drivable_area.ndim == 1:
            # unflattening if flattened
            size = int(sqrt(drivable_area.size(0)))
            drivable_area = drivable_area.view(size, size)

        if flip:
            drivable_area = drivable_area.flip((0, 1))
        # drivable_area is H x W (0.0 <= value <= 1.0)
        H, W = drivable_area.size()

        # convert to pyglet image
        prediction_rgb = create_drivable_area_prediction_image(drivable_area)
        prediction_rgba = torch.cat([
            prediction_rgb,
            torch.full(
                (prediction_rgb.shape[0], prediction_rgb.shape[1], 1),
                255,
                dtype=torch.uint8
            )
        ], dim=-1)

        image_data = bytes(prediction_rgba.flatten().tolist())
        image = ImageData(width=W, height=H, format="RGBA", data=image_data)
        image_size = H

        x_range = viewer.xlim[1] - viewer.xlim[0]
        y_range = viewer.ylim[1] - viewer.ylim[0]
        image_radius = 70 # m
        x_scale = 12 * ((image_size/viewer.width) / (image_radius*2 / x_range)) # HOW TO DO THIS?
        y_scale = 12 * ((image_size/viewer.height) / (image_radius*2 / y_range))

        render_obj = DrivableAreaRenderObject(
            position=position,
            orientation=orientation,
            alpha=self._alpha,
            image=image,
            x_scale=x_scale,
            y_scale=y_scale
        )

        viewer.add(render_obj, persistent=False)


class DrivableAreaRenderObject:

    def __init__(self, position: Tuple[float, float], orientation: float, alpha: float, image: ImageData, x_scale: float = 1.0, y_scale: float = 1.0):
        self.position = position
        self.orientation = orientation
        self.alpha = alpha
        self.image = image
        self.x_scale = x_scale
        self.y_scale = y_scale

    def render(self) -> None:
        gl.glPushMatrix()
        gl.glTranslatef(self.position[0], self.position[1], 0)
        # rotate by 90°, scale down by a factor of 2, flip along the x axis
        gl.glRotatef(np.rad2deg(self.orientation + 0.5 * np.pi), 0.0, 0.0, 1.0)
        gl.glScalef(-self.x_scale, self.y_scale, 1.0)

        # see https://github.com/openai/gym/pull/1928
        gl.glColor4f(1.0, 1.0, 1.0, self.alpha)

        self.image.anchor_x = self.image.width // 2
        self.image.anchor_y = self.image.height // 2
        self.image.blit(0, 0)

        gl.glPopMatrix()


class RenderTrajectoryPredictionPlugin(BaseRenderPlugin):

    def render(self, viewer: T_Viewer, params: RenderParams) -> None:
        assert isinstance(params.data, CommonRoadData)
        trajectory_prediction_container = params.render_kwargs['output'][0]
        trajectory_prediction = trajectory_prediction_container.prediction
        # data = params.data[params.time_step]
        # vehicle_ids = trajectory_prediction_container.vehicle_ids_tensor
        # vehicle_mask = torch.any(data.vehicle.id == vehicle_ids, dim=1)
        # assert params.trajectory_prediction is not None and len(params.trajectory_prediction) > params.time_step

        for t, prediction in enumerate(trajectory_prediction):
            pos = prediction.position.detach().numpy()
            for i in range(pos.shape[0]):
                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=pos[i],
                    radius=0.5,
                    fill_color=Color((0.7 ** t, 0.0, 1 - 0.7 ** t), alpha=0.6),
                    line_width=0.0,
                    border_color=Color((0.7 ** t, 0.0, 1 - 0.7 ** t), alpha=0.6)
                )
