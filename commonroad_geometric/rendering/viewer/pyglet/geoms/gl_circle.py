from typing import Optional

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Angle, T_Position2D
from commonroad_geometric.rendering.viewer.geoms.circle import InterpolatedCircle
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polygon import GlFilledPolygon
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polyline import GlPolyLine


class GlCircle(GlGeom, InterpolatedCircle):

    def __init__(
        self,
        creator: str,
        origin: T_Position2D,
        radius: float,
        start_angle: T_Angle,
        end_angle: T_Angle,
        fill_color: Optional[Color] = None,
        line_color: Optional[Color] = None,
        line_width: float = 1.0,
        resolution: int = 30,
    ):
        super().__init__(creator=creator)
        self._origin = origin
        self._radius = radius
        self._start_angle = start_angle
        self._end_angle = end_angle
        self._resolution = resolution
        circle_points = self.interpolate_circle_points()

        self._inner_circle = None
        if fill_color is not None:
            self._inner_circle = GlFilledPolygon(
                creator=creator,
                vertices=circle_points,
                color=fill_color,
            )

        self._outer_circle = None
        if line_color is not None:
            self._outer_circle = GlPolyLine(
                creator=creator,
                vertices=circle_points,
                is_closed=True,
                color=line_color,
                line_width=line_width
            )

        if self._inner_circle is None and self._outer_circle is None:
            raise ValueError(f"Circle {fill_color=} and {line_color=} are None!"
                             f"Can't draw circle colorless circle, please specify one or both colors")

    def render_geom(self):
        if self._inner_circle is not None:
            self._inner_circle.render()
        if self._outer_circle is not None:
            self._outer_circle.render()

    # Circle properties
    @property
    def origin(self) -> T_Position2D:
        return self._origin

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def start_angle(self) -> T_Angle:
        return self._start_angle

    @property
    def end_angle(self) -> T_Angle:
        return self._end_angle

    @property
    def resolution(self) -> int:
        return self._resolution
