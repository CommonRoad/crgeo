import logging
import os
import warnings
from typing import Optional, TYPE_CHECKING, Tuple, Union, Any

import pyglet
from numpy import cos, sin
import torch
import numpy as np
from scipy import interpolate
from functools import partial

from crgeo.common.class_extensions.auto_hash_mixin import AutoHashMixin
from crgeo.common.class_extensions.auto_repr_mixin import AutoReprMixin
from crgeo.common.geometry.helpers import TWO_PI
from crgeo.rendering.color_utils import T_ColorTuple3
from crgeo.common.geometry.continuous_polyline import ContinuousPolyline
from crgeo.rendering.defaults import ColorTheme
from crgeo.rendering.viewer.utils import Label, Line, ViewTransform, _add_attrs, make_circle, make_polygon, make_polyline

# Checks if rendering is in rendering headless mode, can be activated by 'export PYGLET_HEADLESS=...' in environment
if "PYGLET_HEADLESS" in os.environ:
    pyglet.options["headless"] = True

from pyglet import gl
from pyglet.canvas.xlib import NoSuchDisplayException

if TYPE_CHECKING:
    from pyglet.window import key


logger = logging.getLogger(__name__)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise ValueError('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))


class Viewer2D(AutoReprMixin, AutoHashMixin):
    """
    Offers low-level pyglet rendering functionality such 
    as drawing geometric shapes on the canvas.
    """

    def __init__(
        self,
        width: int,
        height: int,
        theme: ColorTheme,
        caption: str,
        linewidth_multiplier = 1.0,
        display = None,
        minimize: bool = False,
        resizable: bool = False,
        pretty: bool = False,
        smoothing: bool = True,
        transparent_screenshots: bool = True,
        overlay_margin: int = 30,
        override_background_color: Optional[T_ColorTuple3] = None
    ) -> None:
        from pyglet.window import key
        # Disable error checking for increased performance
        pyglet.options['debug_gl'] = False
        display = get_display(display)

        self.linewidth_multiplier = linewidth_multiplier
        try:
            window_kwargs = dict(
                width=int(width),
                height=int(height),
                display=display,
                resizable=resizable,
                caption=caption
            )
            if pretty:
                window_kwargs['config'] = pyglet.gl.Config(sample_buffers=1, samples=4)
            self.window = pyglet.window.Window(**window_kwargs)
            if minimize:
                self.window.minimize()
            self.success = True
        except NoSuchDisplayException as e:
            self.success = False
            warnings.warn(f"Failed to setup 2D Viewer: {repr(e)}")
            return

        self._keys = key.KeyStateHandler()
        self.window.push_handlers(self._keys)
        self.geoms = []
        self.onetime_geoms = []
        self._queued_screenshot_path: Optional[str] = None
        self._queued_set_size: Optional[Tuple[int, int]] = None
        self._skip_frames: int = 0
        self._frame_count: int = 0
        self._xlim: Tuple[float, float]
        self._ylim: Tuple[float, float]
        self.transform = ViewTransform(
            center_position=(0.0, 0.0),
            rotation=0.0,
            viewport_size=(self.width, self.height),
            scale=(1.0, 1.0),
        )
        self.theme = theme
        self.pretty = pretty
        self.smoothing = smoothing
        self.transparent_screenshots = transparent_screenshots
        self.focus_obstacle_idx = 0
        self.overlay_margin = overlay_margin

        if self.smoothing:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.overlay_labels = None
        #TODO: Remove and integrate with overlay labels
        self.overlay_label = pyglet.text.Label(
            '',
            color=(0, 255, 0, 255) if theme == ColorTheme.DARK else (0, 0, 0, 255),
            font_size=10,
            x=self.width - self.overlay_margin, y=self.height - self.overlay_margin,
            anchor_x='right', anchor_y='top', multiline=True, width=350, height=600, dpi=85,
        )
        self.override_background_color = override_background_color
        self.clear_canvas()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self.window.width

    @property
    def height(self) -> int:
        return self.window.height

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def set_size(self, width: int, height: int) -> None:
        self.window.set_size(width=int(width), height=int(height))
        self.transform.set_viewport_size(width=width, height=height)
        self.overlay_label.x = self.width - self.overlay_margin
        self.overlay_label.y = self.height - self.overlay_margin

    @property
    def keys(self) -> 'key.KeyStateHandler':
        return self._keys

    def close(self):
        try:
            self.window.close()
        except:
            pass

    def skip_frames(
        self,
        n: int = 1
    ) -> None:
        self._skip_frames = n

    def screenshot(
        self,
        output_file: str,
        queued: bool = True,
        size: Optional[Tuple[int, int]] = None
    ) -> None:
        if size is not None:
            current_size = self.size
            if queued:
                self._queued_set_size = current_size
            self.set_size(*size)
        if queued:
            self._queued_screenshot_path = output_file
        else:
            self._capture_screenshot(output_file)

    def _capture_screenshot(
        self, 
        output_path: str
    ) -> None:
        from PIL import Image
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(self.height, self.width, 4)

        if self.transparent_screenshots:
            white_mask = np.min(arr[:, :, 0:3], axis=2) == 255
            arr[white_mask, 3] = 0
            arr = arr[::-1, :, :]
        else:
            arr = arr[::-1, :, 0:3]
        im = Image.fromarray(arr)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        im.save(output_path)

    @property
    def xlim(self) -> Tuple[float, float]:
        return self._xlim

    @property
    def ylim(self) -> Tuple[float, float]:
        return self._ylim

    @staticmethod
    def get_bounding_box(
        min_x: float, max_x: float, min_y: float, max_y: float,
        width: float, height: float
    ) -> Tuple[float, float, float, float]:
        # width = max_x - min_x
        # height = max_y - min_y
        aspect_ratio = width / height
        window_aspect_ratio = width / height
        if aspect_ratio > window_aspect_ratio:
            new_height = width / window_aspect_ratio
            delta_height = new_height - height
            min_y -= delta_height * 0.5
            max_y += delta_height * 0.5
        else:
            new_width = height * window_aspect_ratio
            delta_width = new_width - width
            min_x -= delta_width * 0.5
            max_x += delta_width * 0.5
        return min_x, max_x, min_y, max_y

    def fit_into_viewport(
        self,
        min_x: float, max_x: float, min_y: float, max_y: float,
    ) -> Tuple[float, float, float, float]:
        return self.get_bounding_box(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            width=self.width,
            height=self.height
        )

    def set_bounds(self, left: float, right: float, bottom: float, top: float):
        assert right > left and top > bottom
        self._xlim = (left, right)
        self._ylim = (bottom, top)
        scale_x = self.width / (right - left)
        scale_y = self.height / (top - bottom)
        self.transform.scale = scale_x, scale_y
        self.transform.center_position = (left + right) * 0.5, (top + bottom) * 0.5

    def clear_geoms(self) -> None:
        self.geoms = []

    def clear_canvas(self) -> None:
        if self.override_background_color is not None:
            gl.glClearColor(*self.override_background_color, 1.0)
        else:
            if self.theme == ColorTheme.BRIGHT:
                gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            else:
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    def add_geom(self, geom, index: Optional[int] = None):
        if index is None:
            self.geoms.append(geom)
        else:
            self.geoms.insert(index, geom)

    def add(self, geom, index: Optional[int] = None, persistent: bool = False):
        target = self.geoms if persistent else self.onetime_geoms
        if index is None:
            target.append(geom)
        else:
            target.insert(index, geom)

    def pre_render(self) -> bool:
        if self._skip_frames > 0:
            self._skip_frames -= 1
            self._frame_count += 1
            return False
        return True

    def post_render(self) -> None:
        self.onetime_geoms = []
        if self._queued_set_size is not None:
            self.set_size(*self._queued_set_size)
            self._queued_set_size = None

    def render(self) -> None:
        if self._skip_frames > 0:
            self.onetime_geoms = []
            self._queued_screenshot_path = None
            self._queued_set_size = None
            self._skip_frames -= 1
            self._frame_count += 1
            return

        self.clear_canvas()
        if self.theme == ColorTheme.BRIGHT:
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        # reference: https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glHint.xml
        if self.smoothing:
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            if self.pretty:
                gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            else:
                gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_FASTEST)

        for geom in self.geoms:
            if isinstance(geom, pyglet.sprite.Sprite):
                geom.draw()
            else:
                geom.render()
        for geom in self.onetime_geoms:
            if isinstance(geom, pyglet.sprite.Sprite):
                geom.draw()
            else:
                geom.render()

        if self._queued_screenshot_path is not None:
            self._capture_screenshot(self._queued_screenshot_path)
            self._queued_screenshot_path = None
        
        self._frame_count += 1

    def draw_circle(
        self, origin=(0, 0), radius=10, res=30, filled=True, outline=True, linecolor=(0, 0, 0),
        linewidth=1, start_angle=0, end_angle=TWO_PI, index: Optional[int] = None, persistent: bool = False, **kwargs):
        geom = make_circle(origin=origin, radius=radius, res=res, filled=filled, start_angle=start_angle, end_angle=end_angle)
        _add_attrs(geom, kwargs)
        self.add(geom, index=index, persistent=persistent)
        if filled and outline:
            outl = make_circle(origin=origin, radius=radius, res=res, filled=False)
            if linewidth is not None:
                _add_attrs(outl, {'color': linecolor, 'linewidth': linewidth*self.linewidth_multiplier})
            else:
                _add_attrs(outl, {'color': linecolor})
            self.add(outl, index=index, persistent=persistent)
        return geom

    def draw_polygon(self, v, filled=True, index: Optional[int] = None, persistent: bool = False, **kwargs):
        geom = make_polygon(v=v, filled=filled)
        if 'linewidth' in kwargs:
            kwargs['linewidth'] *= self.linewidth_multiplier
        _add_attrs(geom, kwargs)
        self.add(geom, index=index, persistent=persistent)
        return geom

    def draw_rgb_image(
        self,
        data: np.ndarray,
        pos: np.ndarray,
        scale: float = 1.0,
        **kwargs
    ) -> None:
        from pyglet.gl.gl import GLubyte
        import torch
        data = data.numpy() if isinstance(data, torch.Tensor) else data
        if data.ndim == 2:
            data = data[..., None].repeat(3, axis=-1)
        pixels = data.flatten().astype('int').tolist()
        raw_data = (GLubyte * len(pixels))(*pixels)
        image = pyglet.image.ImageData(
            width=data.shape[0],
            height=data.shape[1],
            format='RGB',
            data=raw_data
        )
        sprite = pyglet.sprite.Sprite(image, x=pos[0], y=pos[1])
        sprite.scale = scale
        self.add(sprite, **kwargs)

    def draw_label(self, label='', font_name=None, font_size=4.5, x=(0.0, 0.0), y=(0.0, 0.0), bold=False, anchor_x='center', anchor_y='center', color=(255, 255, 255, 255), italic=False, height=None, width=None, index: Optional[int] = None, persistent: bool = False, **_kwargs):
        kwargs = dict(label=label, font_name=font_name, font_size=font_size, x=x, y=y, bold=bold, anchor_x=anchor_x, anchor_y=anchor_y, color=color, italic=italic, height=height, width=width, **_kwargs)
        geom = Label(**kwargs)
        _add_attrs(geom, kwargs)
        self.add(geom, index=index, persistent=persistent)
        return geom

    def add_label_to_overlays(self, label='', font_name=None, font_size=4.5, x=(0.0, 0.0), y=(0.0, 0.0), bold=False, anchor_x='center', anchor_y='center', color=(255, 255, 255, 255), italic=False, height=None, width=None, multiline=False, **_kwargs):
        attrs = dict(label=label, font_name=font_name, font_size=font_size, x=x, y=y, bold=bold, anchor_x=anchor_x, anchor_y=anchor_y, color=color, italic=italic, height=height, width=width, multiline=multiline, **_kwargs)
        self.overlay_labels.append(Label(**attrs))

    def draw_polyline(self, v, index: Optional[int] = None, persistent: bool = False, **kwargs):
        geom = make_polyline(v=v)
        if 'linewidth' in kwargs:
            kwargs['linewidth'] *= self.linewidth_multiplier
        _add_attrs(geom, kwargs)
        self.add(geom, index=index, persistent=persistent)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.frombuffer(image_data.data, dtype=np.uint8)
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def transform_vertices(self, points, translation, rotation=0.0, scale=1):
        res = []
        rotation = rotation if rotation is not None else 0.0
        for p in points:
            res.append((
                cos(rotation) * p[0] * scale - sin(rotation) * p[1] * scale + translation[0],
                sin(rotation) * p[0] * scale + cos(rotation) * p[1] * scale + translation[1]))
        return res

    def draw_dashed_line(self, start: np.ndarray, end: np.ndarray, spacing: float = 0.7, density: float = 0.3, **kwargs):
        length = np.linalg.norm(end - start)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        dir = np.array([np.cos(angle), np.sin(angle)])
        n = int(length / spacing)
        for i in range(n):
            start_i = start + dir * i/n * length
            end_i = start_i + density * spacing * dir
            self.draw_line(start_i, end_i, **kwargs)

    def draw_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        index: Optional[int] = None,
        persistent: bool = False,
        dashed: bool = False,
        arc: float = 0.0,
        **kwargs
    ):
        if arc != 0.0:
            midpoint = (start + end)/2
            stacked_coordinates = np.stack([start, midpoint, end])
            try:
                straight_polyline = ContinuousPolyline(stacked_coordinates, min_waypoint_distance=0.0)
            except ValueError as e:
                pass # fallback to straight line
            else:
                lateral_deviation = arc*straight_polyline.length
                mid_arclength = straight_polyline.length/2
                arcpoint = midpoint + straight_polyline.get_normal_vector(mid_arclength) * lateral_deviation
                bended_polyline = ContinuousPolyline(
                    np.stack([start, arcpoint, end]), 
                    refinement_steps=3,
                    interpolator=interpolate.CubicSpline
                )
                waypoints = bended_polyline.waypoints
                return self.draw_polyline(
                    waypoints,
                    index=index,
                    persistent=persistent,
                    **kwargs
                )

        if dashed:
            return self.draw_dashed_line(
                start,
                end,
                **kwargs
            )
        geom = Line(start, end)
        if 'linewidth' in kwargs:
            kwargs['linewidth'] *= self.linewidth_multiplier
        _add_attrs(geom, kwargs)
        self.add(geom, index=index, persistent=persistent)
        return geom

    def draw_arrow(self, base, angle, length, scale: float = 0.45, end_offset: float = 1.0, **kwargs):
        if 'linewidth' in kwargs:
            kwargs['linewidth'] *= self.linewidth_multiplier
        TRIANGLE_POLY = ((-1, -1), (1, -1), (0, 1))
        length = max(0.0, length - end_offset)
        head = (base[0] + length * cos(angle), base[1] + length * sin(angle))
        tri = self.transform_vertices(TRIANGLE_POLY, head, angle - np.pi / 2, scale=scale)
        self.draw_line(base, head, **kwargs)
        self.draw_polygon(tri, **kwargs)

    def draw_arrow_from_to(self, start: np.ndarray, end: np.ndarray, **kwargs): # TODO dashed
        length = np.linalg.norm(end - start)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        self.draw_arrow(
            base=start,
            angle=angle,
            length=length,
            **kwargs
        )

    def draw_shape(self, vertices, position=None, angle=None, color=(1, 1, 1), fill_color=None, filled=True, border=True, linewidth=1,
                   label: str = None, label_color=None, label_width=None, label_height=None, font_size=3, index: Optional[int] = None, **kwargs):
        if position is not None:
            poly_path = self.transform_vertices(vertices, position, angle)
        else:
            poly_path = vertices
        if filled:
            self.draw_polygon(poly_path + [poly_path[0]], color=color if fill_color is None else fill_color, index=index, **kwargs)
        if border:
            border_color = (0, 0, 0) if type(border) == bool else border
            self.draw_polyline(poly_path + [poly_path[0]], linewidth=linewidth, color=border_color if filled else color, index=index, **kwargs)
        if label is not None:
            # Draw the label slightly above the vehicle
            self.draw_label(x=position[0], y=position[1] + (1), label=label, color=label_color, height=label_height, width=label_width, font_size=font_size, index=index, **kwargs)

    def __del__(self):
        self.close()
