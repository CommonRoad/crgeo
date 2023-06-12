import unittest
import numpy as np

from commonroad_geometric.common.geometry.helpers import polyline_length, cut_polyline, resample_polyline, \
    cut_polylines_at_identical_segments


class TestPolylineHelpers(unittest.TestCase):

    def test_polyline_length(self):
        line = np.array([
            [ 0, 0 ],
            [ 1, 1 ],
        ])
        self.assertTrue(polyline_length(line) == np.sqrt(2.0))
        line = np.array([ [ 10 ], [ 20 ] ])
        self.assertTrue(polyline_length(line) == 10)
        lines = np.array([
            [
                [ 0, 0 ],
                [ 0, 0 ],
                [ 1, 1 ],
                [ 1, 1 ],
                [ 0, 1 ],

            ],
            [
                [ 0, 0 ],
                [ 3, 0 ],
                [ 3, 3 ],
                [ 0, 3 ],
                [ 0, 0 ],
            ],
            
        ])
        self.assertTrue(np.allclose(polyline_length(lines), np.array([ np.sqrt(2) + 1, 12.0 ])))

    def test_cut_polyline(self):
        polyline = np.array([
            [ 0, 0 ],
            [ 5, 0 ],
            [ 10, 0 ],
        ])
        self.assertTrue(np.allclose(polyline, cut_polyline(polyline, distance=-1.0)))
        self.assertTrue(np.allclose(polyline, cut_polyline(polyline, distance=0.0)))
        self.assertTrue(np.allclose(polyline, cut_polyline(polyline, distance=11.0)))

        polyline = np.array([
            [ 0, 0 ],
            [ 10, 0 ],
        ])
        p1, p2 = cut_polyline(polyline, distance=5.0)
        self.assertTrue(np.allclose(p1, np.array([ [ 0, 0 ], [ 5, 0 ] ])))
        self.assertTrue(np.allclose(p2, np.array([ [ 5, 0 ], [ 10, 0 ] ])))

        polyline = np.array([[ 10 ], [ 20 ]])
        p1, p2 = cut_polyline(polyline, distance=4.0)
        self.assertTrue(np.allclose(p1, np.array([ [ 10 ], [ 14 ] ])))
        self.assertTrue(np.allclose(p2, np.array([ [ 14 ], [ 20 ] ])))

        polyline = np.array([
            [ 0, 0 ],
            [ 1, 0 ],
            [ 2.5, 0 ],
            [ 5, 0 ],
            [ 9, 0 ],
            [ 25, 0 ],
        ], dtype=float)
        distance = [ -5.0, 0.0, 1.2, 1.2, 1.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0 ]
        segments = cut_polyline(polyline, distance=distance)
        segments_groundtruth = [
            np.array([
                [ 0, 0 ],
                [ 1, 0 ],
                [ 1.2, 0 ],
            ]),
            np.array([
                [ 1.2, 0 ],
                [ 1.5, 0 ],
            ]),
            np.array([
                [ 1.5, 0 ],
                [ 2, 0 ],
            ]),
            np.array([
                [ 2, 0 ],
                [ 2.5, 0 ],
                [ 5, 0 ],
            ]),
            np.array([
                [ 5, 0 ],
                [ 9, 0 ],
                [ 10, 0 ],
            ]),
            np.array([
                [ 10, 0 ],
                [ 20, 0 ],
            ]),
            np.array([
                [ 20, 0 ],
                [ 25, 0 ],
            ]),
        ]
        self.assertTrue(all(
            np.allclose(segment, segment_groundtruth)
            for segment, segment_groundtruth in zip(segments, segments_groundtruth)
        ))

    def test_cut_polylines_at_identical_segments(self):
        p1 = np.array([
            [ 0, 0 ],
            [ 5, 0 ],
            [ 10, 0 ],
        ])
        p2 = 0.5 * (p1 + np.array([[ 3.0, 0.0 ]]))
        for dist in [ -1.0, 0.0, 11.0 ]:
            c_p1, c_p2 = cut_polylines_at_identical_segments(lines=[ p1, p2 ], distance=dist)
            self.assertTrue(np.allclose(p1, c_p1))
            self.assertTrue(np.allclose(p2, c_p2))

        p1 = np.array([
            [ 0, 0 ],
            [ 10, 0 ],
        ])
        p2 = np.array([
            [ 1, 2 ],
            [ 19, 4 ],
        ])
        c_p1, c_p2 = cut_polylines_at_identical_segments(lines=[ p1, p2 ], distance=[ 5.0 ])
        self.assertTrue(len(c_p1) == len(c_p2) == 2)
        self.assertTrue(np.allclose(c_p1[0], np.array([ [ 0, 0 ], [ 5, 0 ] ])))
        self.assertTrue(np.allclose(c_p1[1], np.array([ [ 5, 0 ], [ 10, 0 ] ])))
        self.assertTrue(np.allclose(c_p2[0], np.array([ [ 1, 2 ], [ 10, 3 ] ])))
        self.assertTrue(np.allclose(c_p2[1], np.array([ [ 10, 3 ], [ 19, 4 ] ])))

    def test_resample_polyline(self):
        line = np.array([
            [ 0, 0 ],
            [ 5, 0 ],
            [ 12, 0 ],
        ])

        line_groundtruth = np.array([
            [4 * i, 0]
            for i in range(4)
        ])

        line_resampled = resample_polyline(line, interval=4)
        self.assertTrue(np.allclose(line_resampled, line_groundtruth))

        line_resampled = resample_polyline(line, interval=3.0)
        self.assertTrue(np.allclose(line_resampled, line_groundtruth))


if __name__ == "__main__":
    unittest.main()
