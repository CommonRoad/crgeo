from torch import Tensor
import numpy as np
import math
import torch
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