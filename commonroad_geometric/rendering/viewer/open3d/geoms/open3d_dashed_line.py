import numpy as np
from open3d.cuda.pybind.geometry import Geometry, LineSet
from open3d.cuda.pybind.utility import Vector2iVector, Vector3dVector
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Position2D, T_Vertices
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, add_third_dimension


class Open3DDashedLine(Open3DGeom, PolyLine):
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
        vertices = np.array([start, end])
        vertices_3d = add_third_dimension(points=vertices)
        self._line_set = self._create_line_set(
            vertices=vertices_3d,
            spacing=spacing,
        )
        self._o3d_material_record = MaterialRecord()
        self._o3d_material_record.shader = 'unlitLine'
        self._o3d_material_record.line_width = line_width
        self._o3d_material_record.base_color = color.as_rgba()

    @staticmethod
    def _create_line_set(
        vertices: T_Vertices,
        spacing: float,
    ) -> LineSet:
        start = vertices[0]
        end = vertices[-1]

        length = np.linalg.norm(end - start)
        # cosine = np.dot(start, end) / np.linalg.norm(start) / np.linalg.norm(end)
        # angle = np.arccos(np.clip(cosine, -1, 1))
        # line_direction = np.array([np.cos(angle), np.sin(angle)])
        num_line_segments = int(length / spacing)
        # Example: 3 line segments
        # -L1-___-L2-___-L3- where ___ represents the spacing and -Ln- the density
        start_points, step = np.linspace(
            start=start,
            end=end,
            num=num_line_segments,
            endpoint=False,
            retstep=True
        )
        end_points = start_points + step * spacing

        points = np.hstack(
            start_points,
            end_points
        )
        num_points, _ = points.shape
        lines = np.arange(num_points)
        pass
        lines = lines.reshape()
        lines = np.array([0, 1, 2, 3, 4, 5])
        points = Vector3dVector(points)
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
