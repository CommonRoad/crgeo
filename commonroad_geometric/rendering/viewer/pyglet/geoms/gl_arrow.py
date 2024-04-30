from typing import Optional

import numpy as np

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Angle, T_Position2D
from commonroad_geometric.rendering.viewer.geoms.arrow import Arrow
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polygon import GlFilledPolygon
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polyline import GlPolyLine
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, transform_vertices_2d


class GlArrow(GlGeom, Arrow):
    def __init__(
        self,
        creator: str,
        origin: T_Position2D,
        angle: T_Angle,
        length: float,
        line_color: Color,
        arrow_head_color: Optional[Color] = None,
        line_width: float = DEFAULT_LINE_WIDTH,
        arrow_head_size: float = 0.5,
        arrow_head_offset: float = 1.0,
    ):
        r"""
        Wrapper class which compounds the arrow line and head.

        Args:
            creator (str): Creator of the arrow, e.g. __name__ of the render plugin which called draw_arrow.
            origin (T_Position2D): The origin of the arrow.
            angle (T_Angle): The angle of the arrow.
            length (float): The length of the arrow.
            line_color (Color): The color of the line part of the arrow.
            arrow_head_color (Color): The color of the arrow head.
            line_width (float): The width of the line part of the arrow.
            arrow_head_size (float): Relative size of the arrow head in comparison to the line.
            arrow_head_offset (float): Offset of the arrow head from the line.
        """
        super().__init__(creator=creator)
        line_length = max(0.0, length - arrow_head_offset)
        head_start = origin + line_length * np.array([np.cos(angle), np.sin(angle)])
        vertices = np.array([origin, head_start])
        self.arrow_line = GlPolyLine(
            creator=creator,
            vertices=vertices,
            is_closed=False,
            color=line_color,
            line_width=line_width
        )
        base_triangle = np.array([[-1, -1], [1, -1], [0, 1]])
        arrow_head_triangle = transform_vertices_2d(
            vertices=base_triangle,
            translation=head_start,
            rotation=angle - np.pi / 2,
            scale=(arrow_head_size, arrow_head_size)
        )
        self.arrow_head = GlFilledPolygon(
            creator=creator,
            vertices=arrow_head_triangle,
            color=arrow_head_color or line_color
        )

    def render_geom(self):
        self.arrow_line.render()
        self.arrow_head.render()
