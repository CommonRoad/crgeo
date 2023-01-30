from crgeo.rendering.viewer.viewer_2d import Viewer2D
from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin
from crgeo.rendering.types import RenderParams
from crgeo.rendering.color_utils import T_ColorTuple

class RenderLaneletCurvaturePlugin(BaseRendererPlugin):
    def __init__(
        self,
        resolution: int = 25,
        color_positive: T_ColorTuple = (0.2, 0.2, 1.0, 0.5),
        color_negative: T_ColorTuple = (1.0, 0.2, 0.2, 0.5),
        radius_multiplier: float = 2.0,
    ) -> None:
        self.resolution = resolution
        self.color_positive = color_positive
        self.color_negative = color_negative
        self.radius_multiplier = radius_multiplier

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        for lanelet in params.scenario.lanelet_network.lanelets:
            lanelet_path = params.simulation.get_lanelet_center_polyline(lanelet.lanelet_id)

            for t in range(self.resolution):
                arclength = lanelet_path.length * t / (self.resolution + 1)
                pos = lanelet_path(arclength)
                curvature = lanelet_path.get_curvature(arclength)
                radius = abs(curvature) * self.radius_multiplier
                color = self.color_positive if curvature > 0 else self.color_negative
                viewer.draw_circle(
                    origin=pos,
                    radius=radius,
                    color=color,
                    outline=False,
                    linecolor=(0.1, 1.0, 0.1),
                    linewidth=None
                )
