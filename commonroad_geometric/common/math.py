from typing import Optional
import numpy as np


def calc_closest_factors(c: int):
    a, b, i = 1, c, 0
    while a < b:
        i += 1
        if c % i == 0:
            a = i
            b = c//a
    return [b, a]


def scale_feature(
    fmin: Optional[float],
    fmax: Optional[float],
    value: float
) -> float:
    if fmin is None or fmax is None:
        return value
    clipped_value = np.clip(value, fmin, fmax)
    mean = 0.5 * (fmax + fmin) # TODO
    interval = 0.5 * (fmax - fmin)
    scaled_value = (clipped_value - mean) / interval
    return scaled_value
