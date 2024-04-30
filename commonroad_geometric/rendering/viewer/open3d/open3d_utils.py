"""
    This module contains utilities required for Open3D geoms.
"""

import numpy as np

from commonroad_geometric.rendering.types import T_Vertices


def create_line_segment_indices(
    vertices: T_Vertices,
    is_closed: bool
) -> np.ndarray:
    r"""
    Helper function for creating line segment indices for the vertices of a line set.
    Each line segment has a start index and an end index pointing to a specific vertex.

    Args:
        vertices (T_Vertices): The vertices to be connected as line segments
        is_closed (bool): If True, the line segments should form a loop by connecting the last to the first point.

    Returns:
        indices of all line segments
    """
    num_points, dim = vertices.shape
    if num_points <= 1:
        return np.array([])
    first_indices = np.arange(start=0, stop=num_points - 1, dtype=int)
    second_indices = np.arange(start=1, stop=num_points, dtype=int)

    line_segment_indices = np.stack(
        arrays=(first_indices, second_indices),
        axis=-1
    )

    if is_closed:
        last_to_first = np.array([[num_points - 1, 0]], dtype=int)
        line_segment_indices = np.append(
            arr=line_segment_indices,
            values=last_to_first,
            axis=0
        )

    return line_segment_indices
