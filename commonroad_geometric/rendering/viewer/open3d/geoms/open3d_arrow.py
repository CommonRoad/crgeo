from typing import Optional

import numpy as np
from open3d.cuda.pybind.geometry import Geometry, TriangleMesh
from open3d.cuda.pybind.utility import Vector3dVector
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Angle, T_Position2D
from commonroad_geometric.rendering.viewer.geoms.arrow import Arrow
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polygon import Open3DFilledPolygon
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polyline import Open3DPolyLine
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, transform_vertices_2d


class Open3DArrow3D(Open3DGeom, Arrow):
    """This class wraps an arrow created by Open3D."""

    def __init__(
        self,
        creator: str,
        color: Color | list[Color],
        cylinder_radius: float = 1.0,
        cone_radius: float = 1.5,
        cylinder_height: float = 5.0,
        cone_height: float = 4.0,
        resolution: int = 20,
        cylinder_split: int = 4,
        cone_split: int = 1,
    ):
        super().__init__(creator=creator)
        self._arrow_triangle_mesh = TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
            resolution=resolution,
            cylinder_split=cylinder_split,
            cone_split=cone_split
        )
        self._arrow_triangle_mesh: TriangleMesh
        # TODO Add origin, angle, length transformation
        transformation = np.eye(4, dtype=float)
        self._arrow_triangle_mesh.transform(transformation)

        self._o3d_material_record = MaterialRecord()
        self._o3d_material_record.shader = 'unlitLine'
        match color:
            case Color() as color:
                self._o3d_material_record.base_color = color.as_rgba()
            case list() as colors:
                # Base color needs to be white to display multiple colors on one triangle mesh
                white = Color("white")
                self._o3d_material_record.base_color = white.as_rgba()
                self._arrow_triangle_mesh.vertex_colors = Vector3dVector(colors)

    # Open3DGeom properties
    @property
    def o3d_geometries(self) -> list[Geometry]:
        return [self._arrow_triangle_mesh]

    @property
    def o3d_material_records(self) -> list[MaterialRecord]:
        return [self._o3d_material_record]


class Open3DArrow2D(Open3DGeom, Arrow):

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
        self.arrow_line = Open3DPolyLine(
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
        self.arrow_head = Open3DFilledPolygon(
            creator=creator,
            vertices=arrow_head_triangle,
            color=arrow_head_color or line_color
        )

    # Open3DGeom properties
    @property
    def o3d_geometries(self) -> list[Geometry]:
        return [*self.arrow_line.o3d_geometries, *self.arrow_head.o3d_geometries]

    @property
    def o3d_material_records(self) -> list[MaterialRecord]:
        return [*self.arrow_line.o3d_material_records, *self.arrow_head.o3d_material_records]
