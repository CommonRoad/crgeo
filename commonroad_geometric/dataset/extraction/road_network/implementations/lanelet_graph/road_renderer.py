import os
import random
from typing import Dict, Iterable

# https://github.com/pyglet/pyglet/issues/51
if "PYGLET_HEADLESS" in os.environ:
    import pyglet
    pyglet.options["headless"] = True

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as gl

TWO_PI = 2 * np.pi
DEBUG = False

# TODO delete?

class RoadRenderer:

    def __init__(self, size: int, multisampling_samples: int = 0):
        self.size = size
        # display = pyglet.canvas.get_display()
        # screen = display.get_default_screen()
        # config = screen.get_best_config()
        # context = config.create_context(share=None)
        config_kwargs = {
            "buffer_size": 24,  # RGB, no alpha
            "depth_size": 0,  # depth buffer not needed
        }
        if multisampling_samples > 0:
            config_kwargs["sample_buffers"] = 1
            config_kwargs["samples"] = multisampling_samples
        config = gl.Config(**config_kwargs)

        self.window = pyglet.window.Window(
            # context=context,
            config=config,
            vsync=False,
            width=size,
            height=size,
        )
        gl.glViewport(0, 0, size, size)

        buffer_manager = pyglet.image.get_buffer_manager()
        self.color_buffer = buffer_manager.get_color_buffer()

    def _set_up_projection_matrix(self, scale: float, rotation_deg: float, translation: np.ndarray) -> None:
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.size, -self.size // 2, self.size // 2, -1.0, 1.0)
        gl.glScalef(scale, scale, scale)
        gl.glRotatef(-rotation_deg, 0.0, 0.0, -1.0)
        gl.glTranslatef(-translation[0], -translation[1], 0)

    def _clear_window(self):
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)  # | gl.GL_DEPTH_BUFFER_BIT

    def render_road_coverage(
            self,
            *,
            graph: nx.DiGraph,
            node_ids: Iterable,
            scale: float,
            lanelet_orientation: float,
            lanelet_position: np.ndarray,
    ) -> np.ndarray:
        self._set_up_projection_matrix(scale=scale, rotation_deg=-np.rad2deg(lanelet_orientation), translation=lanelet_position)
        self._clear_window()

        # draw lanelets
        for node_id in node_ids:
            # prepare QUAD_STRIP vertices
            node = graph.nodes[node_id]
            num_vertices = node["left_vertices"].shape[0]
            vertices = np.empty((2 * num_vertices, 2), dtype=float)
            vertices[::2] = node["right_vertices"]
            vertices[1::2] = node["left_vertices"]

            if DEBUG:
                gl.glColor3f(1.0, random.random(), random.random())  # only the first color channel will be extracted
            else:
                gl.glColor3f(1.0, 1.0, 1.0)
            pyglet.graphics.draw(vertices.shape[0], gl.GL_QUAD_STRIP, ("v2f", vertices.ravel(order="C")))

        gl.glFlush()
        self.window.flip()

        road_coverage = np.frombuffer(self.color_buffer.get_image_data().get_data(), dtype=np.uint8)
        road_coverage = road_coverage.reshape((self.size, self.size, 4))[:, :, 0].astype(dtype=float) / 255.0
        assert np.all((road_coverage == 0.0) | (road_coverage == 1.0))
        # road_coverage dimensions: [y, x]

        return road_coverage

    def render_road_orientation(
        self,
        *,
        graph: nx.DiGraph,
        node_ids: Iterable,
        scale: float,
        lanelet_orientation: float,
        lanelet_position: np.ndarray,
        lanelet_orientation_buckets: int = 8,
        node_segments_orientations: Dict[int, np.ndarray],
    ) -> np.ndarray:
        self._set_up_projection_matrix(scale=scale, rotation_deg=-np.rad2deg(lanelet_orientation), translation=lanelet_position)

        road_orientation = np.empty((lanelet_orientation_buckets, self.size, self.size), dtype=float)
        # dimensions: [orientation, y, x]

        bucket_size = TWO_PI / lanelet_orientation_buckets
        nodes = []
        for node_id in node_ids:
            segment_orientations = node_segments_orientations[node_id]
            segment_orientations = (segment_orientations + (TWO_PI - lanelet_orientation)) % TWO_PI
            segment_orientation_buckets = (segment_orientations / bucket_size).astype(int)
            nodes.append((graph.nodes[node_id], segment_orientation_buckets))

        for orientation in range(lanelet_orientation_buckets):
            self._clear_window()
            # draw all lanelet segments which have an orientation within the current orientation range
            for node, segment_orientation_buckets in nodes:
                segments = (segment_orientation_buckets == orientation).astype(int)
                num_segments = segments.sum()
                if num_segments == 0:
                    continue

                # prepare QUADS vertices
                left_vertices, right_vertices = node["left_vertices"], node["right_vertices"]
                vertices = np.empty((4 * num_segments, 2), dtype=float)
                segment_indices = np.nonzero(segments)[0]
                vertices[0::4] = right_vertices[segment_indices]
                vertices[1::4] = left_vertices[segment_indices]
                vertices[2::4] = left_vertices[segment_indices + 1]
                vertices[3::4] = right_vertices[segment_indices + 1]

                if DEBUG:
                    a = (orientation + 0.5) * bucket_size
                    gl.glColor3f(1.0, 0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a))  # only the first color channel will be extracted
                else:
                    gl.glColor3f(1.0, 1.0, 1.0)
                pyglet.graphics.draw(vertices.shape[0], gl.GL_QUADS, ("v2f", vertices.ravel(order="C")))

            gl.glFlush()

            road_coverage = np.frombuffer(self.color_buffer.get_image_data().get_data(), dtype=np.uint8)
            road_coverage = road_coverage.reshape((self.size, self.size, 4))[:, :, 0].astype(dtype=float) / 255.0
            assert np.all((road_coverage == 0.0) | (road_coverage == 1.0))
            # road_coverage dimensions: [y, x]
            road_orientation[orientation] = road_coverage

        return road_orientation

    def close(self):
        self.window.close()
