from __future__ import annotations

from typing import Optional, Tuple, Type, Union

import numpy as np
import shapely
from shapely.geometry.linestring import LineString
from commonroad.geometry.transform import rotate_translate
from scipy import interpolate
from shapely.geometry.base import BaseGeometry

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin


class ContinuousPolyline(AutoReprMixin):
    @classmethod
    def merge(cls, *polylines: Union[ContinuousPolyline, np.ndarray]) -> ContinuousPolyline:
        waypoints = np.vstack([l.waypoints if isinstance(l, ContinuousPolyline) else l for l in polylines])
        return cls(waypoints)

    def __init__(
        self,
        waypoints: Union[np.ndarray,LineString],
        # TODO: Are there defaults reasonable?
        waypoint_resolution: int = 200,
        linestring_resolution: int = 200,
        extrapolate: bool = False,
        refinement_steps: int = 3,
        min_waypoint_distance: float = 1e-3,
        interpolator: Type[interpolate.CubicHermiteSpline] = interpolate.PchipInterpolator
    ) -> None:
        if waypoints.shape[0] < 2:
            raise ValueError("Need at least 2 waypoints")

        if isinstance(waypoints, LineString):
            waypoints = np.array(waypoints.coords)

        waypoints = np.vstack([
            np.expand_dims(waypoints[0], 0),
            waypoints[1:][np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > min_waypoint_distance]
        ])
        self.init_waypoints = waypoints.copy()
        for _ in range(refinement_steps):
            # Calculating arclengths
            diff = np.diff(waypoints, axis=0)
            delta_arc = np.sqrt(np.sum(diff ** 2, axis=1))
            self._arclengths = np.concatenate([[0], np.cumsum(delta_arc)])

            path_coords = interpolator(x=self._arclengths, y=waypoints, axis=0, extrapolate=extrapolate)
            waypoints = path_coords(np.linspace(self._arclengths[0], self._arclengths[-1], waypoint_resolution))

        self._waypoint_resolution = waypoint_resolution
        self._linestring_resolution = linestring_resolution
        self._num_points = int(linestring_resolution * self.length)
        self._waypoints = waypoints.copy()
        self._path_coords = path_coords
        path_derivatives = path_coords.derivative()  # TODO: Only compute on first access
        path_dderivatives = path_derivatives.derivative()
        self._path_derivatives = path_derivatives
        self._path_dderivatives = path_dderivatives

        self._linspace = np.linspace(0, self.length, self._num_points)
        self._coordinate_arr = self._path_coords(self._linspace)
        self._derivative_arr = self._path_derivatives(self._linspace)
        self._tangential_dir_arr = np.arctan2(self._derivative_arr[:, 1], self._derivative_arr[:, 0])  # TODO: Only compute on first access
        self._linestring = shapely.geometry.LineString(self._coordinate_arr)

        self._curvature_arr: Optional[np.ndarray] = None
        self._normal_arr: Optional[np.ndarray] = None

        self._start_direction = self.get_direction(0)
        self._end_direction = self.get_direction(self.length)

    @property
    def length(self) -> float:
        """Length of path in meters."""
        return self._arclengths[-1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._coordinate_arr.shape

    @property
    def waypoints(self) -> np.ndarray:
        """Waypoint used for path."""
        return self._waypoints

    @property
    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.waypoints[:, 0], self.waypoints[:, 1]

    @property
    def start_direction(self) -> float:
        """Direction at path's starting point."""
        return self._start_direction

    @property
    def end_direction(self) -> float:
        """Direction at path's endpoint."""
        return self._end_direction

    @property
    def start(self) -> np.ndarray:
        """Coordinates of the path's starting point."""
        return self._path_coords(0)

    @property
    def end(self) -> np.ndarray:
        """Coordinates of the path's end point."""
        return self._path_coords(self.length)

    def slice(self, from_arclength: float, to_arclength: float, inclusive: bool = True) -> np.ndarray:
        assert to_arclength > from_arclength
        start_index = self._get_cache_index(from_arclength)
        end_index =  self._get_cache_index(to_arclength)
        return self._coordinate_arr[start_index:end_index + int(inclusive)]

    def __call__(self, arclength: float, use_cached: bool = False) -> np.ndarray:
        """
        Returns the (x,y) point corresponding to the
        specified arclength.

        Returns
        -------
        point : np.array
        """
        arclength = min(self.length, max(0, arclength))
        if use_cached:
            return self.get_cached_pos(arclength)
        else:
            return self._path_coords(arclength)

    def get_cached_pos(self, arclength: float) -> np.ndarray:
        index = self._get_cache_index(arclength)
        return self._coordinate_arr[index]

    def _get_cache_index(self, arclength: float) -> int:
        index = min(self._num_points - 1, int(self._linestring_resolution * arclength))
        return index

    def get_direction(self, arclength: float, use_cached: bool = True, clamp: bool = True) -> float:
        """
        Returns the direction in radians with respect to the
        positive x-axis.

        Args:
            arclength (float): Arclength position on the path in meters.
            use_cached (bool, optional): Whether to lookup precomputed value. Defaults to True.

        Returns:
            float: Direction of the path at the specified point
        """
        if clamp:
            arclength = max(0.0, min(arclength, self.length))
        if use_cached:
            index = self._get_cache_index(arclength)
            return self._tangential_dir_arr[index]
        derivative = self._path_derivatives(arclength)
        return np.arctan2(derivative[..., 1], derivative[..., 0])

    def get_curvature(self, arclength: float) -> float:
        curvature_arr = self.get_curvature_arr()
        index = self._get_cache_index(arclength)
        return curvature_arr[index]

    def get_curvature_arr(self) -> np.ndarray:
        if self._curvature_arr is None:
            dd_arr = self._path_dderivatives(self._linspace)
            xd = self._derivative_arr[:, 0]
            yd = self._derivative_arr[:, 1]
            xdd = dd_arr[:, 0]
            ydd = dd_arr[:, 1]
            self._curvature_arr = (xd*ydd - yd*xdd) / np.power(xd** 2 + yd** 2, 1.5)
        return self._curvature_arr

    def get_normal_vector(self, arclength) -> np.ndarray:
        if self._normal_arr is None:
            self._normal_arr = self._compute_normal_arr()
        index = self._get_cache_index(arclength)
        return self._normal_arr[index]

    def get_projected_direction(
        self,
        position: np.ndarray,
        use_cached: bool = True
    ) -> float:
        arclength = self.get_projected_arclength(position)
        return self.get_direction(arclength, use_cached=use_cached)

    def get_projected_arclength(
        self,
        position: np.ndarray,
        relative: bool = False,
    ) -> float:
        """
        Returns the arc length value corresponding to the point
        on the path which is closest to the specified position.

        Returns
        -------
        arclength : float
        """
        arclength = self._linestring.project(shapely.geometry.Point(position))
        arclength = max(0.0, min(arclength, self.length))
        if relative:
            return arclength / self.length
        return arclength

    def get_projected_position(self, position: np.ndarray) -> np.ndarray:
        """
        Projects position to polyline.

        Returns
        -------
        projection : np.ndarray
        """
        arclength = self.get_projected_arclength(position)
        projection = self(arclength)
        return projection

    def get_projected_distance(
        self,
        position: np.ndarray,
        arclength: Optional[float] = None,
        use_cached: bool = True
    ) -> float:
        """
        Returns the distance of the given position to the point
        on the path which is closest to the specified position.

        Returns
        -------
        arclength : float
        """
        if arclength is None:
            arclength = self.get_projected_arclength(position)
        if arclength > self.length:
            projection = self(self.length)
        elif arclength < 0:
            projection = self(0)
        else:
            projection = self(arclength, use_cached=use_cached)
        distance = float(np.linalg.norm(position - projection))
        return distance

    def get_lateral_distance(self, position: np.ndarray) -> float:
        arclength = self.get_projected_arclength(position)
        xy_diff_position = position - self(arclength)
        lateral_distance = rotate_translate(
            xy_diff_position[None, :],
            [0, 0],
            -self.get_direction(arclength)
        ).squeeze(0)[1]
        return lateral_distance

    def lateral_translate(self, distance: float) -> ContinuousPolyline:
        if self._normal_arr is None:
            self._normal_arr = self._compute_normal_arr()
        new_waypoints = self._coordinate_arr + distance*self._normal_arr
        return type(self)(new_waypoints)

    def lateral_translate_point(self, arclength: float, lateral_distance: float) -> np.ndarray:
        if self._normal_arr is None:
            self._normal_arr = self._compute_normal_arr()
        index = self._get_cache_index(arclength)
        point = self._coordinate_arr[index] + lateral_distance*self._normal_arr[index]
        return point

    def intersect(self, other: 'ContinuousPolyline') -> BaseGeometry:
        return self._linestring.intersection(other._linestring)

    def _compute_normal_arr(self) -> np.ndarray:
        return np.column_stack([self._derivative_arr[:, 1], -self._derivative_arr[:, 0]])
