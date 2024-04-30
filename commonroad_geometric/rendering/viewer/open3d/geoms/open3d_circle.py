from typing import Optional

from open3d.cuda.pybind.geometry import Geometry
from open3d.cuda.pybind.visualization.rendering import MaterialRecord

from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.types import T_Angle, T_Position2D
from commonroad_geometric.rendering.viewer.geoms.circle import InterpolatedCircle
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polygon import Open3DFilledPolygon
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polyline import Open3DPolyLine


class Open3DCircle(Open3DGeom, InterpolatedCircle):

    def __init__(
        self,
        creator: str,
        origin: T_Position2D,
        radius: float,
        start_angle: T_Angle,
        end_angle: T_Angle,
        fill_color: Optional[Color] = None,
        line_color: Optional[Color] = None,
        line_width: float = 1.0,
        resolution: int = 30,
    ):
        super().__init__(creator=creator)
        self._origin = origin
        self._radius = radius
        self._start_angle = start_angle
        self._end_angle = end_angle
        self._resolution = resolution
        circle_points = self.interpolate_circle_points()

        self._inner_circle = None
        if fill_color is not None:
            self._inner_circle = Open3DFilledPolygon(
                creator=creator,
                vertices=circle_points,
                color=fill_color,
            )

        self._outer_circle = None
        if line_color is not None:
            self._outer_circle = Open3DPolyLine(
                creator=creator,
                vertices=circle_points,
                is_closed=True,
                color=line_color,
                line_width=line_width
            )

        if self._inner_circle is None and self._outer_circle is None:
            raise ValueError(f"Circle {fill_color=} and {line_color=} are None!"
                             f"Can't draw circle colorless circle, please specify one or both colors")

    # Open3DGeom properties
    @property
    def o3d_geometries(self) -> list[Geometry]:
        o3d_geometries = []
        if self._inner_circle is not None:
            o3d_geometries.extend(self._inner_circle.o3d_geometries)
        if self._outer_circle is not None:
            o3d_geometries.extend(self._outer_circle.o3d_geometries)
        return o3d_geometries

    @property
    def o3d_material_records(self) -> list[MaterialRecord]:
        o3d_material_records = []
        if self._inner_circle is not None:
            o3d_material_records.extend(self._inner_circle.o3d_material_records)
        if self._outer_circle is not None:
            o3d_material_records.extend(self._outer_circle.o3d_material_records)
        return o3d_material_records

    # Circle properties
    @property
    def origin(self) -> T_Position2D:
        return self._origin

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def start_angle(self) -> T_Angle:
        return self._start_angle

    @property
    def end_angle(self) -> T_Angle:
        return self._end_angle

    @property
    def resolution(self) -> int:
        return self._resolution
