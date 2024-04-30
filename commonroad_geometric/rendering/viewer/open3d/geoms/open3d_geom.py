from abc import abstractmethod

from open3d.cuda.pybind.geometry import Geometry
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering.viewer.geoms.base_geom import BaseGeom


class Open3DGeom(BaseGeom):
    r"""
    Interface for Open3D specific behavior for the BaseGeom class.
    Open3D is a more high level framework, providing most functionality and attributes by default.
    """

    @property
    @abstractmethod
    def o3d_geometries(self) -> list[Geometry]:
        ...

    @property
    @abstractmethod
    def o3d_material_records(self) -> list[MaterialRecord]:
        ...
