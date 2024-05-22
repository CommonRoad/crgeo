"""Module containing utils for rendering."""

import numpy as np
from scipy.spatial import Delaunay

from commonroad_geometric.rendering.types import T_Position2D, T_Position3D, T_Vertices

DEFAULT_LINE_WIDTH = 1


def add_third_dimension(
    points: T_Vertices,
    pad_value: int | float = 0.0
) -> T_Vertices:
    r"""
    Adds third dimension with pad_value to points.

    Args:
        points (T_Vertices): The points which should be embedded into the third dimension.
        pad_value (int | float): The value additional dimensions are padded with. Defaults to 0.0.

    Returns:
        3-dimensional points
    """
    num_points, dim = points.shape
    assert dim <= 3, (f'Vertices need to be a "1, 2 or 3-dimensional" array with shape (number_of_points, dimension<=3),'
                      f'got ({num_points=}, {dim=})')
    new_points = np.full((num_points, 3 - dim), fill_value=pad_value)
    points_3d = np.hstack((points, new_points))
    return points_3d


def transform_vertices_2d(
    vertices: T_Vertices,
    translation: T_Position2D = (0.0, 0.0),
    rotation: float = 0.0,
    scale: T_Position2D = (1.0, 1.0),
) -> T_Vertices:
    r"""
    First rotates, then translates, then scales the given vertices.

    Args:
        vertices (T_Vertices): 2D vertices
        translation (T_Position2D): The x, y translation vector
        rotation (float): The counterclockwise rotation angle in radians
        scale (T_Position2D): The x, y scaling vector

    Returns:
        transformed 2D vertices

    References:
        https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation
    """
    num_points, dim = vertices.shape
    assert dim == 2, 'Vertices need to be a "2-dimensional" array with shape (number_of_points, dimension=2)'
    translation = np.asarray(translation)
    scale = np.asarray(scale)
    rotation = -float(rotation)
    rotation_transform = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])

    rotated_vertices = vertices @ rotation_transform
    translated_vertices = rotated_vertices + translation
    scaled_vertices = translated_vertices * scale
    return scaled_vertices


def transform_vertices_3d(
    vertices: T_Vertices,
    translation: T_Position3D = (0.0, 0.0, 0.0),
    rotation: T_Position3D = (0.0, 0.0, 0.0),
    scale: T_Position3D = (1.0, 1.0, 1.0),
) -> T_Vertices:
    r"""
    First translates, then scales, then rotates the given vertices.

    Args:
        vertices (T_Vertices): 3D vertices
        translation (T_Position3D): The x, y, z translation vector
        rotation (float): The counterclockwise rotation angles in radians about the x, y, z axis as a vector
        scale (T_Position3D): The x, y, z scaling vector

    Returns:
        transformed 3D vertices

    References:
        https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    # Convert to homogeneous coordinates
    num_points, dim = vertices.shape
    assert dim == 3, 'Vertices need to be a "3-dimensional" array with shape (number_of_points, dimension=3)'
    homogeneous_vertices = np.hstack((
        vertices,
        np.ones(shape=(num_points, 1), dtype=vertices.dtype)
    ))

    rotation_x, rotation_y, rotation_z = rotation
    x_rotation_transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(rotation_x), -np.sin(rotation_x), 0.0],
        [0.0, np.sin(rotation_x), np.cos(rotation_x), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    y_rotation_transform = np.array([
        [np.cos(rotation_y), 0.0, -np.sin(rotation_y), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [np.sin(rotation_y), 0.0, np.cos(rotation_y), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    z_rotation_transform = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0.0, 0.0],
        [np.sin(rotation_z), np.cos(rotation_z), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    scale_x, scale_y, scale_z = scale
    scale_transform = np.array([
        [scale_x, 0.0, 0.0, 0.0],
        [0.0, scale_y, 0.0, 0.0],
        [0.0, 0.0, scale_z, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    x, y, z = translation
    translation_transform = np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])
    rotation_transform = x_rotation_transform @ y_rotation_transform @ z_rotation_transform
    # 1. Translate, 2. Scale, 3. Rotate
    affine_transform = rotation_transform @ scale_transform @ translation_transform
    # Apply affine transformation
    transformed_homogeneous_vertices = affine_transform @ homogeneous_vertices.T
    # Convert back to inhomogeneous coordinates
    transformed_vertices = transformed_homogeneous_vertices[:-1, :].T
    return transformed_vertices


def triangulate(vertices: T_Vertices) -> T_Vertices:
    num_points, dim = vertices.shape
    if num_points == 1:
        return np.array([0, 0])
    if num_points == 2:
        return np.array([0, 1])

    # Triangulate the polygon
    # 'QJ' ensures that flat objects (e.g. all z-coordinates 0) can be triangulated
    # "- use 'QJ'  to joggle the input and make it full dimensional"
    triangulation = Delaunay(vertices, qhull_options='QJ')
    return triangulation.simplices
