import numpy as np
from crgeo.common.geometry.helpers import TWO_PI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def relative_angles(angle_1: torch.Tensor, angle_2: torch.Tensor) -> torch.Tensor:
    """Computes the angle between two angles."""

    phi = (angle_2 - angle_1) % TWO_PI
    bigger_than_pi = phi > np.pi
    output = phi.new_zeros(size=phi.shape)
    output[bigger_than_pi] = phi[bigger_than_pi] - TWO_PI
    output[~bigger_than_pi] = phi[~bigger_than_pi]

    return output