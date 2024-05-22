from pyglet import gl

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Vertices
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.pyglet.attr.line_width import LineWidth
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, add_third_dimension


class GlPolyLine(GlGeom, PolyLine):
    def __init__(
        self,
        creator: str,
        vertices: T_Vertices,
        is_closed: bool,
        color: Color | list[Color],
        line_width: float = DEFAULT_LINE_WIDTH
    ) -> None:
        r"""
        Creates a polyline consisting of vertices that are connected to each other in the order given in the array.

        Args:
            creator (str): Creator of the polyline, e.g. __name__ of the render plugin which called draw_polyline.
            vertices (T_Vertices): 2D or 3D array of vertices.
            is_closed (bool): If True, the last and first point are connected to form a closed polyline.
            color (Color | list[Color]): Either one color which is applied to all line segments or a list of colors with
                                         as many elements as there are line segments, where each color is applied to the
                                         respective segment in the order given by the list.
            line_width (float): The line width of the polyline.
        """
        super().__init__(creator=creator)
        self._vertices = vertices
        self._is_closed = is_closed
        self._color = color
        self._line_width = LineWidth(stroke=line_width)

    def render_geom(self):
        gl.glColor4f(*self._color.as_rgba())
        gl.glBegin(gl.GL_LINE_LOOP if self._is_closed else gl.GL_LINE_STRIP)

        vertices_3d = add_third_dimension(self.vertices)
        match self._color:
            case Color() as color:
                # Draw all vertices with one color
                gl.glColor4f(*color.as_rgba())
                for vertex in vertices_3d:
                    gl.glVertex3f(*vertex)
            case list() as colors:
                # Draw each vertex with a new color
                for vertex, color in zip(vertices_3d, colors):
                    gl.glColor4f(*color.as_rgba())
                    gl.glVertex3f(*vertex)
        gl.glEnd()

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
