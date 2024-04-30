import numpy as np

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Position2D
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.pyglet.attr.line_width import LineWidth
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH


class GlDashedLine(GlGeom, PolyLine):
    def __init__(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        color: Color,
        line_width: float = DEFAULT_LINE_WIDTH,
        spacing: float = 0.5,
    ):
        r"""
        Creates a dashed line consisting of vertices that are connected to each other in the order given in the array.

        Args:
            creator (str): Creator of the dashed line, e.g. __name__ of the render plugin which called draw_dashed_line.
            start (T_Position2D): 2D start point.
            end (T_Position2D): 2D end point.
            color (Color): Color of the line.
            line_width (float): The line width of each line segment.
            spacing (float): The space between each line segment.
        """
        super().__init__(creator=creator)
        self._vertices = np.array([start, end])
        self._color = color
        self._line_width = LineWidth(stroke=line_width)
        self._spacing = spacing

    def render_geom(self):
        raise NotImplementedError

    # PolyLine properties
    @property
    def vertices(self):
        return self._vertices

    @property
    def start(self):
        return self.vertices[0]

    @property
    def end(self):
        return self.vertices[-1]
