import warnings
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import math
from shapely.geometry import LineString

TWO_PI = 2.0 * math.pi
T_FloatOrArray = TypeVar("T_FloatOrArray", float, np.ndarray)


def relative_orientation(angle_1: T_FloatOrArray, angle_2: T_FloatOrArray) -> T_FloatOrArray:
    """Computes the angle between two angles."""

    phi = (angle_2 - angle_1) % TWO_PI
    if isinstance(angle_1, np.ndarray):
        phi[phi > np.pi] -= TWO_PI
    else:
        if phi > np.pi:
            phi -= TWO_PI
    return phi


def make_valid_orientation(angle: float) -> float:
    angle = angle % TWO_PI
    if np.pi <= angle <= TWO_PI:
        angle = angle - TWO_PI
    assert -np.pi <= angle <= np.pi
    return angle

def princip(angle: float) -> float:
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def chaikins_corner_cutting(coords: np.ndarray, refinements=1):
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def rotate_2d_matrix(rotation: float, dtype=float) -> np.ndarray:
    cos, sin = math.cos(rotation), math.sin(rotation)
    return np.array([
        [ cos, -sin ],
        [ sin, cos ],
    ], dtype=dtype)


def rotate_2d(x: np.ndarray, r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([
        c * x[0] - s * x[1],
        s * x[0] + c * x[1],
    ])


def translate_rotate_2d(x: np.ndarray, t: np.ndarray, r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([
        c * x[0] - s * x[1] + c * t[0] - s * t[1],
        s * x[0] + c * x[1] + s * t[0] + c * t[1],
    ])


def scale_2d(scale: float, dtype=float) -> np.ndarray:
    return np.array([
        [ scale, 0 ],
        [ 0, scale ],
    ], dtype=dtype)


def affine_2d_homogeneous(rotation: float = 0.0, scale: float = 1.0, translation: Optional[np.ndarray] = None,
                          dtype=float) -> np.ndarray:
    if translation is None:
        translation = np.array([ 0, 0 ], dtype=dtype)
    cos, sin = np.cos(rotation), np.sin(rotation)
    return np.array([
        [ scale * cos, scale * -sin, translation[0] ],
        [ scale * sin, scale * cos, translation[1] ],
        [ 0, 0, 1 ],
    ], dtype=dtype)
    

def polyline_length(line: np.ndarray) -> np.ndarray:
    """Compute the Euclidean length of one or more polylines."""
    assert line.ndim >= 2 and line.shape[-2] > 1
    delta_dist = np.sqrt(np.sum(np.diff(line, axis=-2) ** 2, axis=-1))
    return np.sum(delta_dist, axis=-1)


def _cut_polyline(line: np.ndarray, distance: Union[List[float], np.ndarray]) \
        -> Tuple[List[np.ndarray], List[Tuple[int, float]]]:
    assert line.ndim == 2 and line.shape[0] > 1
    coords = line.astype(dtype=float, copy=True)

    delta_dist = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    cumulative_dist = np.cumsum(delta_dist, axis=0)
    i = 0
    segments, cutting_points = [], []
    dist_along_line = 0
    while coords.shape[0] > i + 1 and len(distance) > 0:
        if distance[0] <= dist_along_line:
            # Skip negative distances and segments of zero length
            distance = distance[1:]
            continue

        if cumulative_dist[i] < distance[0]:
            # Next cutting point is beyond the current line segment
            dist_along_line = cumulative_dist[i]
            i += 1
            continue

        if cumulative_dist[i] == distance[0]:
            # Next cutting point is exactly at the next point on the line
            segments.append(coords[:i+2].copy())
            cutting_points.append((i + 1, 0.0))
            dist_along_line = cumulative_dist[i]
            i += 1

        else:  # distance[0] < cumulative_dist[i]
            # Next cutting point is between the current and next point on the line
            alpha = (distance[0] - dist_along_line) / (cumulative_dist[i] - dist_along_line)
            cut_point = (1 - alpha) * coords[i] + alpha * coords[i + 1]
            segment = np.vstack([
                coords[:i+1],
                cut_point,
            ])
            segments.append(segment)
            cutting_points.append((i, alpha))
            dist_along_line = distance[0]
            coords[i] = cut_point

        distance = distance[1:]
        coords = coords[i:]
        cumulative_dist = cumulative_dist[i:]
        i = 0
    
    if coords.shape[0] >= 2:
        # Add the remaining line segment
        segments.append(coords)

    return (segments or [ coords ]), cutting_points


def cut_polyline(line: Union[np.ndarray, LineString],
                 distance: Union[float, List[float], np.ndarray]) -> List[np.ndarray]:
    """Cut a polyline into multiple segments at one or more distances along the polyline."""
    if isinstance(line, LineString):
        coords = np.array(line.coords)
    else:
        assert line.ndim == 2 and line.shape[0] > 1
        coords = line.astype(dtype=float, copy=True)
    if isinstance(distance, float):
        distance = [ distance ]
    polyline_segments, _ = _cut_polyline(line=coords, distance=distance)
    return polyline_segments


def cut_polylines_at_identical_segments(
    lines: List[Union[np.ndarray, LineString]],
    distance: Union[float, List[float], np.ndarray]
) -> List[List[np.ndarray]]:
    lines = [
        np.array(line.coords, dtype=float) if isinstance(line, LineString) else line.astype(dtype=float, copy=True)
        for line in lines
    ]
    if isinstance(distance, float):
        distance = [ distance ]
    assert lines[0].shape[0] > 1
    assert all(line.ndim == 2 and line.shape[0] == lines[0].shape[0] for line in lines)
    segments_first_line, cutting_points = _cut_polyline(line=lines[0], distance=distance)

    cut_lines = [ segments_first_line ] + [ [] for _ in range(len(lines) - 1) ]
    for i_segment, alpha in cutting_points:
        for i_lines in range(1, len(lines)):
            if alpha == 0.0:
                segment = lines[i_lines][:i_segment + 1]
            else:
                p = (1 - alpha) * lines[i_lines][i_segment] + alpha * lines[i_lines][i_segment + 1]
                segment = np.vstack([ lines[i_lines][:i_segment + 1], p ])
                lines[i_lines][i_segment] = p
            lines[i_lines] = lines[i_lines][i_segment:]
            cut_lines[i_lines].append(segment)
    for i_lines in range(1, len(lines)):
        segment = lines[i_lines]
        cut_lines[i_lines].append(segment)
    return cut_lines


def resample_polyline(line: Union[np.ndarray, LineString], interval: Union[float, int]) -> LineString:
    if isinstance(line, np.ndarray):
        line = LineString(line)
    if isinstance(interval, int):
        n_points = interval
    else:
        n_points = int(line.length // interval)

    waypoints = np.linspace(0, line.length, n_points)
    points = [line.interpolate(x) for x in waypoints]

    # assert np.allclose(np.array(line)[-1], np.array(points[-1]))
    # assert np.allclose(np.array(line)[0], np.array(points[0]))
    # assert len(points) == n_points

    return LineString(points)
