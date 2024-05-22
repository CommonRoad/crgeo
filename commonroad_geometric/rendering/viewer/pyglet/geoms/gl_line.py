import numpy as np
from pyglet import gl

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Position2D
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.pyglet.attr.line_width import LineWidth
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH


class GlLine(GlGeom, PolyLine):

    def __init__(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        color: Color,
        line_width: float = DEFAULT_LINE_WIDTH
    ):
        r"""
        Creates a line from start to end.

        Args:
            creator (str): Creator of the polyline, e.g. __name__ of the render plugin which called draw_polyline.
            start (T_Position2D): The 2D start point.
            end (T_Position2D): The 2D end point.
            color (Color): The color of the line.
            line_width (float): The width of the line.
        """
        super().__init__(creator=creator, attrs=[LineWidth(line_width)])
        self._start = start
        self._end = end
        self._color = color

    def render_geom(self):
        gl.glColor4f(*self._color.as_rgba())
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()

    @property
    def vertices(self):
        return np.array([self.start, self.end])

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end
