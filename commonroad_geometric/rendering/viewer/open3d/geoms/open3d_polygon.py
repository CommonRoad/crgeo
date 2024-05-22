import numpy as np
from open3d.cuda.pybind.geometry import Geometry, TetraMesh, TriangleMesh
from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector, Vector4iVector
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Vertices
from commonroad_geometric.rendering.viewer.geoms.polygon import FilledPolygon
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.utils import add_third_dimension, triangulate


class Open3DFilledPolygon(Open3DGeom, FilledPolygon):

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
        self._triangle_mesh = self._create_triangle_mesh(vertices=vertices)
        self._o3d_material_record = MaterialRecord()
        self._o3d_material_record.shader = 'unlitLine'

        match color:
            case Color() as color:
                self._o3d_material_record.base_color = color.as_rgba()
            case list() as colors:
                # Base color needs to be white to display multiple colors on one triangle mesh
                white = Color("white")
                self._o3d_material_record.base_color = white.as_rgba()
                self._triangle_mesh.vertex_colors = Vector3dVector(colors)

    @staticmethod
    def _create_triangle_mesh(vertices: T_Vertices) -> TetraMesh:
        # [Open3D WARNING] Geometry type 10 is not supported yet! -> TetraMesh
        # tetra_mesh = TetraMesh()
        # vector = Vector3dVector(vertices)
        # tetra_mesh.vertices = vector
        # tetras = triangulate(vertices)
        # tetra_mesh.tetras = Vector4iVector(tetras)
        # return tetra_mesh
        triangles = triangulate(vertices)
        vertices_3d = add_third_dimension(vertices)
        triangle_mesh = TriangleMesh(
            vertices=Vector3dVector(vertices_3d),
            triangles=Vector3iVector(triangles)
        )
        return triangle_mesh

    # Open3DGeom properties
    @property
    def o3d_geometries(self) -> list[Geometry]:
        return [self._triangle_mesh]

    @property
    def o3d_material_records(self) -> list[MaterialRecord]:
        return [self._o3d_material_record]

    # FilledPolygon properties
    @property
    def vertices(self) -> T_Vertices:
        return np.asarray(self._triangle_mesh.vertices)
