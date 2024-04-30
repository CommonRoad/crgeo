from dataclasses import dataclass
import math
from typing import Tuple
import numpy as np
import torch
from pyglet import gl
from pyglet.image import ImageData

from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D


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

@dataclass
class RenderDrivableAreaPlugin(BaseRenderPlugin):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        super().__init__()

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if not isinstance(viewer, GLViewer2D):
            return

        data = params.data

        try:
            if data.ego.drivable_area.ndim == 2 and data.ego.drivable_area.shape[0] == 1:
                image_size = int(math.sqrt(data.ego.drivable_area[0].shape[-1]))
                drivable_area = data.ego.drivable_area.reshape(1, image_size, image_size)[0]
            else:
                drivable_area = data.ego.drivable_area[0]
        except AttributeError:
            return 
        
        if params.ego_vehicle is not None:
            prediction_rgb = (drivable_area * 255).type(torch.uint8).cpu()[:, :, None].repeat(1, 1, 3)
            prediction_rgba = torch.cat([
                prediction_rgb,
                torch.full(
                    (prediction_rgb.shape[0], prediction_rgb.shape[1], 1),
                    255,
                    dtype=torch.uint8
                )
            ], dim=-1)

            image_data = bytes(prediction_rgba.flatten().tolist())

            image = ImageData(width=image_size, height=image_size, format="RGBA", data=image_data)

            x_range = viewer.xlim[1] - viewer.xlim[0]
            y_range = viewer.ylim[1] - viewer.ylim[0]
            image_radius = 70 # m
            x_scale = 12 * ((image_size/viewer.width) / (image_radius*2 / x_range)) # HOW TO DO THIS?
            y_scale = 12 * ((image_size/viewer.height) / (image_radius*2 / y_range))

            render_obj = DrivableAreaRenderObject(
                position=data.ego.pos[0],
                orientation=data.ego.orientation[0],
                alpha=self.alpha,
                image=image,
                x_scale=x_scale,
                y_scale=y_scale
            )
            viewer.add(render_obj, persistent=False)
        else:
            # drawing drivable area of first vehicle
            viewer.draw_rgb_image(
                data=255*drivable_area,
                pos=data.v.pos[0],
                scale=0.2
            )
