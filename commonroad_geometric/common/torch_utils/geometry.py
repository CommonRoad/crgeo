from torch import Tensor
import numpy as np
import math
import torch
from typing import Optional
from commonroad_geometric.common.geometry.helpers import TWO_PI


def relative_angles(angle_1: Tensor, angle_2: Tensor) -> Tensor:
    """Computes the angle between two angles."""

    phi = (angle_2 - angle_1) % TWO_PI
    bigger_than_pi = phi > np.pi
    output = phi.new_zeros(size=phi.shape)
    output[bigger_than_pi] = phi[bigger_than_pi] - TWO_PI
    output[~bigger_than_pi] = phi[~bigger_than_pi]

    return output


def rotate_2d(x: Tensor, r: float) -> Tensor:
    c, s = math.cos(r), math.sin(r)
    return torch.tensor([
        c * x[0] - s * x[1],
        s * x[0] + c * x[1],
    ])


def translate_rotate_2d(x: Tensor, t: Tensor, r: float) -> Tensor:
    c, s = math.cos(r), math.sin(r)
    return torch.tensor([
        c * x[0] - s * x[1] + c * t[0] - s * t[1],
        s * x[0] + c * x[1] + s * t[0] + c * t[1],
    ])



def contains_any_rotated_rectangles(
    x: Tensor, 
    y: Tensor, 
    cx: Tensor, 
    cy: Tensor, 
    width: Tensor, 
    height: Tensor, 
    angle: Tensor, 
    weight: Optional[Tensor] = None,
    reduce: bool = True
):
    """
    Vectorized evaluation of whether a point is contained by at least one rotated rectangle.
    Parameters:
        x (tensor): x-coordinates of the points.
        y (tensor): y-coordinates of the points.
        cx (tensor): x-coordinates of the centers of the rectangles.
        cy (tensor): y-coordinates of the centers of the rectangles.
        width (tensor): widths of the rectangles.
        height (tensor): heights of the rectangles.
        angle (tensor): angles of rotation of the rectangles in radians.
        weight (tensor): if not None, weights for all vehicles
    Returns:
        A tensor of shape x.shape that contains 1 for points inside at least one rectangle and 0 for points outside.
    """

    width = width.squeeze(-1)
    height = height.squeeze(-1)
    angle = angle.squeeze(-1)

    # Translate points to the coordinate system centered on each rectangle
    x = x[..., None]
    y = y[..., None]
    x = x - cx
    y = y - cy
    # Rotate points by negative angle
    cos_theta = torch.cos(-angle - 0.5*np.pi)
    sin_theta = torch.sin(-angle - 0.5*np.pi)
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    # Check if points are inside any rectangle
    left = -width / 2
    right = width / 2
    bottom = -height / 2
    top = height / 2
    is_inside = (x_rot > left) & (x_rot < right) & (y_rot > bottom) & (y_rot < top)
    if weight is not None:
        is_inside = is_inside * weight

    if reduce:
        is_inside = is_inside.sum(dim=-1)

    return is_inside