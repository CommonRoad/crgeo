import unittest

import numpy as np
import open3d as o3d

from commonroad_geometric.rendering.viewer.open3d.open3d_utils import create_line_segment_indices
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polygon import Open3DFilledPolygon
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polyline import Open3DPolyLine
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_line import Open3DDashedLine, Open3DLine


def compare_arrays(array1, array2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    flattened_array1 = array1.flatten()
    flattened_array2 = array2.flatten()
    return np.allclose(flattened_array1, flattened_array2)


class TestCreateLines(unittest.TestCase):
    def test_create_lines_no_loop(self):
        vertices = [[0, 0], [0, 1], [1, 1], [1, 0]]
        expected_lines = [[0, 1], [1, 2], [2, 3]]
        self.assertEqual(create_line_segment_indices(vertices, is_closed=False), expected_lines)

    def test_create_lines_loop(self):
        vertices = [[0, 0], [0, 1], [1, 1], [1, 0]]
        expected_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.assertEqual(create_line_segment_indices(vertices, is_closed=True), expected_lines)

    def test_create_lines_empty_input(self):
        vertices = []
        expected_lines = []
        self.assertEqual(create_line_segment_indices(vertices, is_closed=False), expected_lines)
        self.assertEqual(create_line_segment_indices(vertices, is_closed=True), expected_lines)

    def test_create_lines_one_vertex(self):
        vertices = [[0, 0]]
        expected_lines = []
        self.assertEqual(create_line_segment_indices(vertices, is_closed=False), expected_lines)
        self.assertEqual(create_line_segment_indices(vertices, is_closed=True), expected_lines)

    def test_create_lines_two_vertices(self):
        vertices = [[0, 0], [0, 1]]
        expected_lines = [[0, 1]]
        self.assertEqual(create_line_segment_indices(vertices, is_closed=False), expected_lines)
        expected_lines_loop = [[0, 1], [1, 0]]
        self.assertEqual(create_line_segment_indices(vertices, is_closed=True), expected_lines_loop)


class TestPolyLine(unittest.TestCase):
    def setUp(self):
        self.vertices = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        self.lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.polyline = Open3DPolyLine(self.vertices, close=True, colors=self.colors)
        self.lineset = self.polyline.o3d_geometries

    def test_init(self):
        self.assertIsInstance(self.lineset, o3d.geometry.LineSet)
        self.assertTrue(compare_arrays(self.lineset.points, self.vertices))
        self.assertTrue(compare_arrays(self.lineset.colors, self.colors))

    def test_set_linewidth(self):
        self.polyline.set_linewidth(5)
        self.assertEqual(self.polyline.o3d_material_records.line_width, 5)

    def test_set_color(self):
        self.polyline.set_colorColor((0.5, 0.5, 0.5, 0.5))
        self.assertTrue(compare_arrays(self.polyline.o3d_material_records.base_color, Color((0.5, 0.5, 0.5, 0.5))))

    def test_set_colors(self):
        new_colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]
        self.polyline.set_colors(new_colors)
        self.assertTrue(compare_arrays(self.polyline.o3d_geometries.colors, new_colors))


class TestLine(unittest.TestCase):
    def setUp(self):
        self.start = (1, 2, 3)
        self.end = (4, 5, 6)
        self.colors = Color((0.1, 0.2, 0.3, 1))
        self.line = Open3DLine(start=self.start, end=self.end, colors=self.colors)

    def test_render1(self):
        lineset = self.line.render_geom()
        self.assertTrue(compare_arrays(np.asarray(lineset.points), [self.start, self.end]))
        self.assertTrue(compare_arrays(lineset.lines, [[0, 1]]))

    def test_set_linewidth(self):
        new_linewidth = 5
        self.line.set_linewidth(new_linewidth)
        self.assertEqual(self.line.linewidth, new_linewidth)

    def test_enable_linewidth(self):
        self.line.enable_linewidth(True)
        self.assertEqual(self.line.o3d_material_records.line_width, self.line.linewidth)
        self.line.enable_linewidth(False)
        self.assertEqual(self.line.o3d_material_records.line_width, DEFAULT_LINE_WIDTH)

    def test_set_color(self):
        r, g, b, a = 0.5, 0.2, 0.9, 0.7
        self.line.set_color(r, g, b, a)
        self.assertTrue(compare_arrays(self.line.o3d_material_records.base_color, (r, g, b, a)))


class TestDashedLine(unittest.TestCase):

    def setUp(self):
        self.start = (1, 2, 3)
        self.end = (4, 5, 6)
        self.linewidth = 0.5
        self.spacing = 0.7
        self.density = 0.3
        self.colors = Color((0, 0, 0, 1.0))
        self.dashed_line = Open3DDashedLine(start=self.start, end=self.end, linewidth=self.linewidth,
                                            spacing=self.spacing, density=self.density, colors=self.colors)

    def test_render1(self):
        start_2d = np.array(self.start[0:2])
        end_2d = np.array(self.end[0:2])
        length = np.linalg.norm(end_2d - start_2d)
        angle = np.arctan2(end_2d[1] - start_2d[1], end_2d[0] - start_2d[0])
        dir = np.array([np.cos(angle), np.sin(angle)])
        n = int(length / self.spacing)
        expected_points = []
        for i in range(n):
            start_i = start_2d + dir * i / n * length
            end_i = start_i + self.density * self.spacing * dir
            start_i_3d = np.append(start_i, [0])
            end_i_3d = np.append(end_i, [0])
            expected_points.append([start_i_3d, end_i_3d])

        linesets = self.dashed_line.render_geom()
        for lineset, points in zip(linesets, expected_points):
            self.assertTrue(compare_arrays(np.asarray(lineset.points), points))
        self.assertIsInstance(self.dashed_line.render_geom(), list)
        self.assertIsInstance(self.dashed_line.render_geom()[0], o3d.geometry.LineSet)

    def test_set_linewidth(self):
        new_linewidth = 0.7
        self.dashed_line.set_linewidth(new_linewidth)
        self.assertEqual(self.dashed_line.linewidth, new_linewidth)

    def test_enable_linewidth(self):
        dashed_line = Open3DDashedLine()
        dashed_line.set_linewidth(0.5)
        self.assertEqual(dashed_line.o3d_material_records.line_width, DEFAULT_LINE_WIDTH)
        dashed_line.enable_linewidth(True)
        self.assertEqual(dashed_line.o3d_material_records.line_width, 0.5)
        dashed_line.enable_linewidth(False)
        self.assertEqual(dashed_line.o3d_material_records.line_width, DEFAULT_LINE_WIDTH)

    def test_set_color(self):
        dashed_line = Open3DDashedLine()
        dashed_line.set_color(0.5, 0.5, 0.5)
        self.assertTrue(compare_arrays(dashed_line.o3d_material_records.base_color, Color((0.5, 0.5, 0.5, 1.0))))


class TestFilledPolygon(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        color = Color((255, 0, 0, 1))
        self.polygon = Open3DFilledPolygon(vertices=self.vertices, colors=color)

    def test_init(self):
        self.assertTrue(compare_arrays(self.polygon.vertices, self.vertices))
        self.assertTrue(compare_arrays(self.polygon.o3d_material_records.base_color, Color((1.0, 0.0, 0.0, 1.0))))

    def test_shift_color(self):
        color = Color((255, 0, 0, 1))
        shifted_color = self.polygon._shift_color(color)
        self.assertTrue(compare_arrays(shifted_color, Color((1.0, 0.0, 0.0, 1.0))))

    def test_shift_colors(self):
        colors = [Color((255, 0, 0, 1)), Color((0, 255, 0, 1)), Color((0, 0, 255, 1))]
        shifted_colors = self.polygon._shift_colors(colors)
        self.assertTrue(
            compare_arrays(shifted_colors, [Color((1.0, 0.0, 0.0, 1.0)), Color((0.0, 1.0, 0.0, 1.0)), Color((0.0, 0.0, 1.0, 1.0))]))

    def area_of_polygon(self, x, y):
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def areas_equal(self, vertices):
        triangles = _triangulate(vertices)
        x = [vertices[i][0] for i in range(len(vertices))]
        y = [vertices[i][1] for i in range(len(vertices))]
        total_area = self.area_of_polygon(x, y)
        triangles_area = 0
        for t in triangles:
            x_t = [vertices[t[i]][0] for i in range(3)]
            y_t = [vertices[t[i]][1] for i in range(3)]
            triangles_area += self.area_of_polygon(x_t, y_t)
        self.assertAlmostEqual(triangles_area, total_area)

    def test_triangulate_area(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.areas_equal(vertices)

    def test_triangulate_edgecase_colinear_vertices(self):
        vertices = [(0, 0), (1, 0), (2, 0), (3, 0)]
        triangles = _triangulate(vertices)
        self.assertEqual(triangles, None)

    def test_triangulate_edgecase_concave_polygon(self):
        vertices = [(0, 0), (1, 0), (2, 1), (1, 2), (0, 2)]
        self.areas_equal(vertices)

    def test_triangulate_edgecase_convex_polygon(self):
        vertices = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
        self.areas_equal(vertices)

    def test_triangulate_concave(self):
        # Test with concave input
        vertices = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        triangles = _triangulate(vertices)
        self.assertIsNotNone(triangles)

    def test_triangulate_less_than_3_vertices(self):
        # Test with less than 3 vertices
        vertices = [[0, 0], [1, 0]]
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        with self.assertRaises(ValueError):
            _triangulate(vertices)

    def test_triangulate_assertion(self):
        # Test with invalid input shape (not 2D)
        vertices = [[0, 0], [1, 0], [1, 1], [0, 1], [2]]
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        with self.assertRaises(AssertionError):
            _triangulate(vertices)

    def test_triangulate_2D(self):
        # Test with valid 2D input
        vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        triangles = _triangulate(vertices)
        self.assertIsNotNone(triangles)

    def test_triangulate_3D(self):
        # Test with valid 3D input
        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        triangles = _triangulate(vertices)
        self.assertIsNotNone(triangles)

    def test_triangulate_empty_input(self):
        # Test with empty input
        vertices = []
        filled_polygon = Open3DFilledPolygon(vertices=vertices, colors=Color((1, 1, 1, 1)))
        with self.assertRaises(ValueError):
            triangles = _triangulate(vertices)

    def test_render_geom(self):
        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        color = Color((255, 0, 0, 1))
        polygon = Open3DFilledPolygon(vertices=vertices, colors=color)
        tri_mesh = polygon.render_geom()
        self.assertIsInstance(tri_mesh, o3d.geometry.TriangleMesh)
        self.assertTrue(compare_arrays(tri_mesh.vertices, vertices))
        self.assertTrue(tri_mesh.triangles, [(0, 1, 2), (0, 2, 3)])


if __name__ == '__main__':
    unittest.main()
