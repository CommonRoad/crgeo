import numpy as np
from open3d.cuda.pybind.geometry import Geometry, LineSet
from open3d.cuda.pybind.utility import Vector2iVector, Vector3dVector
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Vertices
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.open3d.open3d_utils import create_line_segment_indices
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, add_third_dimension


class Open3DPolyLine(Open3DGeom, PolyLine):
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
        self._line_set = self._create_line_set(
            vertices=vertices,
            is_closed=is_closed
        )
        self._o3d_material_record = MaterialRecord()
        self._o3d_material_record.shader = 'unlitLine'
        self._o3d_material_record.line_width = line_width

        match color:
            case Color() as color:
                self._o3d_material_record.base_color = color.as_rgba()
            case list() as colors:
                self._line_set.colors = Vector3dVector(colors)

    @staticmethod
    def _create_line_set(
        vertices: T_Vertices,
        is_closed: bool,
    ) -> LineSet:
        vertices_3d = add_third_dimension(points=vertices)
        points = Vector3dVector(vertices_3d)
        lines = create_line_segment_indices(
            vertices=vertices_3d,
            is_closed=is_closed
        )
        lines = Vector2iVector(lines)
        line_set = LineSet(points=points, lines=lines)
        return line_set

    # Open3DGeom properties
    @property
    def o3d_geometries(self) -> list[Geometry]:
        return [self._line_set]

    @property
    def o3d_material_records(self) -> list[MaterialRecord]:
        return [self._o3d_material_record]

    # PolyLine properties
    @property
    def vertices(self):
        return np.asarray(self._line_set.points)

    @property
    def start(self):
        return self.vertices[0]

    @property
    def end(self):
        return self.vertices[-1]
