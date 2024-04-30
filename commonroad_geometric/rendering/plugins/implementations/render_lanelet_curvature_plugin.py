from dataclasses import dataclass
from typing import Optional

from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderLaneletCurvaturePlugin(BaseRenderPlugin):
    resolution: int = 25
    color_positive: Optional[Color] = Color((0.2, 0.2, 1.0), alpha=0.5)
    color_negative: Optional[Color] = Color((1.0, 0.2, 0.2), alpha=0.5)
    radius_multiplier: float = 2.0

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        for lanelet in params.scenario.lanelet_network.lanelets:
            lanelet_path = params.simulation.get_lanelet_center_polyline(lanelet.lanelet_id)

            for t in range(self.resolution):
                arclength = lanelet_path.length * t / (self.resolution + 1)
                position = lanelet_path(arclength)
                curvature = lanelet_path.get_curvature(arclength)
                radius = abs(curvature) * self.radius_multiplier
                color = self.color_positive if curvature > 0 else self.color_negative
                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=position,
                    radius=radius,
                    fill_color=color,
                    border_color=Color((0.1, 1.0, 0.1)),
                )
