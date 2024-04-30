import numpy as np

from commonroad_geometric.rendering.viewer.utils import add_third_dimension, transform_vertices_2d, transform_vertices_3d


def test_add_third_dimension_1d():
    points = np.array([[1], [2], [3]])
    points_3d = add_third_dimension(points=points)
    expected_points_3d = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    assert np.allclose(points_3d, expected_points_3d)


def test_add_third_dimension_2d():
    points_2d = np.array([[1, 2], [2, 3]])
    points_3d = add_third_dimension(
        points=points_2d,
        pad_value=5
    )
    expected_points_3d = np.array([[1, 2, 5], [2, 3, 5]])
    assert np.allclose(points_3d, expected_points_3d)


def test_add_third_dimension_3d():
    points_2d = np.array([[1, 2, 3]])
    points_3d = add_third_dimension(
        points=points_2d,
        pad_value=3
    )
    expected_points_3d = np.array([[1, 2, 3]])
    assert np.allclose(points_3d, expected_points_3d)


def test_transform_vertices_2d():
    vertices = np.array([[0, 0], [1, 0], [1, 1]])

    translation = np.array([1.0, -1.0])
    translated_vertices = transform_vertices_2d(
        vertices=vertices,
        translation=translation
    )
    expected_translated_vertices = vertices + translation
    assert np.allclose(translated_vertices, expected_translated_vertices)

    rotation = np.pi
    rotated_vertices = transform_vertices_2d(
        vertices=vertices,
        rotation=rotation
    )
    expected_rotated_vertices = -vertices
    assert np.allclose(rotated_vertices, expected_rotated_vertices)

    scale = (2.0, 5.0)
    scaled_vertices = transform_vertices_2d(
        vertices=vertices,
        scale=scale
    )
    expected_scale_vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 5.0]])
    assert np.allclose(scaled_vertices, expected_scale_vertices)

    transformed_vertices = transform_vertices_2d(
        vertices=vertices,
        translation=translation,
        rotation=np.pi,
        scale=scale
    )
    expected_transformed_vertices = np.array([[2.0, -5.0], [0.0, -5.0], [0.0, -10.0]])
    assert np.allclose(transformed_vertices, expected_transformed_vertices)


def test_transform_vertices_3d():
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    translation = np.array([1.0, -1.0, 2.0])
    translated_vertices = transform_vertices_3d(
        vertices=vertices,
        translation=translation
    )
    expected_translated_vertices = vertices + translation
    assert np.allclose(translated_vertices, expected_translated_vertices)

    # Rotate 180 degrees around the x-axis
    rotation = (np.pi, 0.0, 0.0)
    rotated_vertices = transform_vertices_3d(
        vertices=vertices,
        rotation=rotation
    )
    expected_rotated_vertices = np.vstack((vertices[0, :], -vertices[1:, :]))
    assert np.allclose(rotated_vertices, expected_rotated_vertices)

    scale = np.array([2.0, 5.0, 3.0])
    scaled_vertices = transform_vertices_3d(
        vertices=vertices,
        scale=scale
    )
    expected_scaled_vertices = vertices * scale
    assert np.allclose(scaled_vertices, expected_scaled_vertices)

    transformed_vertices = transform_vertices_3d(
        vertices=vertices,
        translation=translation,
        rotation=rotation,
        scale=scale
    )
    expected_transformed_vertices = np.array([
        [4.0, 5.0, -6.0],
        [2.0, -0.0, -6.0],
        [2.0, 5.0, -9.0]
    ])
    assert np.allclose(transformed_vertices, expected_transformed_vertices)
