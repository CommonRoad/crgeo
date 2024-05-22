from pyglet import gl

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Vertices
from commonroad_geometric.rendering.viewer.geoms.polygon import FilledPolygon
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.utils import add_third_dimension


class GlFilledPolygon(GlGeom, FilledPolygon):

    def __init__(
        self,
        creator: str,
        vertices: T_Vertices,
        color: Color | list[Color]
    ):
        r"""
        Creates a polygon given by its vertices.

        Args:
            creator (str): Creator of the circle, e.g. __name__ of the render plugin which called draw_polygon.
            vertices (T_Vertices): 2D or 3D array of vertices of the polygon.
            color (Color | list[Color]): Either one color which is applied to the whole polygon or a list of colors with
                                         as many elements as there are vertices, where each color is applied to the
                                         respective vertex in the order given by the list.
        """
        super().__init__(creator=creator)
        self._vertices = vertices
        self._color = color

    def render_geom(self):
        num_points, dim = self.vertices.shape
        if num_points == 4:
            gl.glBegin(gl.GL_QUADS)
        elif num_points > 4:
            gl.glBegin(gl.GL_POLYGON)
        else:
            gl.glBegin(gl.GL_TRIANGLES)

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

    # FilledPolygon properties
    @property
    def vertices(self) -> T_Vertices:
        return self._vertices
