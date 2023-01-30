from typing import Optional
from dataclasses import dataclass

from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin
from crgeo.rendering.types import RenderParams
from crgeo.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderDrivableAreaStyle():
    ...


class RenderDrivableAreaPlugin(BaseRendererPlugin):
    def __init__(
        self,
        style: Optional[RenderDrivableAreaStyle] = None,
        **kwargs
    ) -> None:
        self.style = style if style is not None else RenderDrivableAreaStyle(**kwargs)

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:

        data = params.data

        if params.ego_vehicle is not None:
            viewer.draw_rgb_image(
                data=255*data.ego.drivable_area,
                pos=data.ego.pos,
                scale=0.2
            )
        else:
            # drawing drivable area of first vehicle
            viewer.draw_rgb_image(
                data=255*data.v.drivable_area[0],
                pos=data.v.pos[0],
                scale=0.2
            )
