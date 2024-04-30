import numpy as np

from commonroad_geometric.common.geometry.helpers import TWO_PI
from commonroad_geometric.rendering.viewer.geoms.circle import interpolate_circle_points


def test_interpolate_circle_points():
    circle_points = interpolate_circle_points(
        origin=(10, 10),
        radius=25,
        start_angle=0,
        end_angle=TWO_PI,
        resolution=50
    )
    assert circle_points.shape == (51, 2)
    first_point = circle_points[0, :]
    last_point = circle_points[-1, :]
    assert (first_point == np.array([35., 10.])).all()
    assert np.allclose(first_point, last_point)
