from abc import ABC, abstractmethod

import numpy as np

from commonroad_geometric.rendering.types import T_Angle, T_Position2D


def interpolate_circle_points(
    origin: T_Position2D,
    radius: float,
    start_angle: T_Angle,
    end_angle: T_Angle,
    resolution: int,
) -> np.ndarray:
    r"""
    Interpolates points along a circle given:

    Args:
        origin (T_Position2D): Origin/center of the circle.
        radius (float): Radius of the circle.
        start_angle (T_Angle): Starting angle to draw the perimeter point.
        end_angle (T_Angle): End angle to draw the perimeter point.
        resolution (int): Resolution of the circle - number of points representing perimeter of the circle.

    Returns:
        2D-Array of interpolated circle points, np.ndarray with shape (resolution + 1, 2)
    """
    angle_dt = (end_angle - start_angle) / resolution
    x, y = origin
    # x_coords = np.cos(angled_indices) * radius + x
    # TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead.
    # x and y could be a numpy.float64
    x, y = float(x), float(y)

    angled_indices = start_angle + np.arange(resolution + 1) * angle_dt
    x_coords = np.cos(angled_indices) * radius + x
    y_coords = np.sin(angled_indices) * radius + y
    points = np.stack((x_coords, y_coords), axis=1)
    return points


class InterpolatedCircle(ABC):
    @property
    @abstractmethod
    def origin(self) -> T_Position2D:
        ...

    @property
    @abstractmethod
    def radius(self) -> float:
        ...

    @property
    @abstractmethod
    def start_angle(self) -> T_Angle:
        ...

    @property
    @abstractmethod
    def end_angle(self) -> T_Angle:
        ...

    @property
    @abstractmethod
    def resolution(self) -> int:
        ...

    def interpolate_circle_points(self) -> np.ndarray:
        r"""
        Returns:
            2D-Array of interpolated circle points, np.ndarray with shape (2, resolution + 1)
        """
        return interpolate_circle_points(
            origin=self.origin,
            radius=self.radius,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            resolution=self.resolution
        )
