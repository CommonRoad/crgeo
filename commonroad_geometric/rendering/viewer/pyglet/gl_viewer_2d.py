import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pyglet
from pyglet import gl
from pyglet.canvas.xlib import NoSuchDisplayException
from pyglet.window import key
from torch import Tensor

from commonroad_geometric.common.class_extensions.auto_hash_mixin import AutoHashMixin
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.geometry.helpers import TWO_PI
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import CameraView2D
from commonroad_geometric.rendering.types import T_Angle, T_Position2D, T_Vertices
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer, ViewerOptions
from commonroad_geometric.rendering.viewer.pyglet.attr.view_transform import ViewTransform
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_arrow import GlArrow
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_circle import GlCircle
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_dashed_line import GlDashedLine
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_geom import GlGeom
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_label import GlLabel
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_line import GlLine
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polygon import GlFilledPolygon
from commonroad_geometric.rendering.viewer.pyglet.geoms.gl_polyline import GlPolyLine
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, transform_vertices_2d

if "PYGLET_HEADLESS" in os.environ:
    # Checks if rendering is in rendering headless mode
    # Can be activated by 'export PYGLET_HEADLESS=...' in environment
    pyglet.options["headless"] = True

pyglet.options['debug_gl'] = False  # Disable error checking for increased performance

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


@dataclass
class GLViewerOptions(ViewerOptions):
    is_resizable: bool = False
    is_pretty: bool = False
    enable_smoothing: bool = True
    transparent_screenshots: bool = False
    overlay_margin: int = 30
    font_size: int = 14
    custom_background_color: Optional[Color] = None  # TODO Implement as custom theme


class GLViewer2D(BaseViewer[GLViewerOptions, GlGeom], AutoReprMixin, AutoHashMixin):
    """
    Offers low-level pyglet rendering functionality such 
    as drawing geometric shapes on the canvas.
    """

    def __init__(
        self,
        options: GLViewerOptions,
        display: Any = None,
    ) -> None:
        super(GLViewer2D, self).__init__(options=options)
        from pyglet.window import key
        display = get_display(display)

        try:
            window_kwargs = dict(
                width=int(options.scaled_window_width),
                height=int(options.scaled_window_height),
                display=display,
                resizable=options.is_resizable,
                caption=options.caption
            )
            if options.is_pretty:
                window_kwargs['config'] = pyglet.gl.Config(sample_buffers=1, samples=4)
            self.window = pyglet.window.Window(**window_kwargs)
            if options.minimize_window:
                self.window.minimize()
            self.is_initialized = True
        except NoSuchDisplayException as e:
            self.is_initialized = False
            warnings.warn(f"Failed to setup 2D Viewer: {repr(e)}")
            return

        self._keys = key.KeyStateHandler()
        self.window.push_handlers(self._keys)

        self.transform = ViewTransform(
            center_position=(0.0, 0.0),
            rotation=0.0,
            viewport_size=(self.width, self.height),
            scale=(1.0, 1.0),
        )
        self._queued_screenshot_path: Optional[str] = None
        self._queued_set_size: Optional[Tuple[int, int]] = None  # ?

        if options.enable_smoothing:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self._clear_canvas()

    @property
    def is_active(self) -> bool:
        return self.window._active

    @property
    def is_minimized(self) -> bool:
        return self.options.minimize_window

    @property
    def width(self) -> int:
        return self.window.width

    @property
    def height(self) -> int:
        return self.window.height

    @property
    def xlim(self) -> Tuple[float, float]:
        return self._xlim

    @property
    def ylim(self) -> Tuple[float, float]:
        return self._ylim

    @property
    def keys(self) -> 'key.KeyStateHandler':
        return self._keys

    def set_view(self, camera_view: CameraView2D):
        center_x, center_y = camera_view.center_position

        left = center_x - camera_view.view_range / 2
        right = center_x + camera_view.view_range / 2
        bottom = center_y - camera_view.view_range / 2
        top = center_y + camera_view.view_range / 2

        assert right > left and top > bottom
        self._xlim = (left, right)
        self._ylim = (bottom, top)

        scale_x = self.width / (right - left)
        scale_y = self.height / (top - bottom)

        self.transform.scale = scale_x, scale_y
        self.transform.center_position = camera_view.center_position
        self.transform.rotation = camera_view.orientation

    # TODO: can probably be combined with add_geom?
    def add(self, geom, index: Optional[int] = None, persistent: bool = False):
        target = self.geoms if persistent else self.onetime_geoms
        if index is None:
            target.append(geom)
        else:
            target.insert(index, geom)

    def pre_render(self) -> None:
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()
        gl.glViewport(0, 0, self.width, self.height)
        self.transform.enable()

    def _clear_canvas(self) -> None:
        if self.options.custom_background_color is not None:
            gl.glClearColor(*self.options.custom_background_color.as_rgba())
        else:
            if self.options.theme == ColorTheme.BRIGHT:
                gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            else:
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    def render(
        self,
        screenshot_path: Optional[Path] = None
    ) -> None:
        self._clear_canvas()
        if self.options.theme == ColorTheme.BRIGHT:
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        # reference: https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glHint.xml
        if self.options.enable_smoothing:
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            if self.options.is_pretty:
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

        if screenshot_path is not None:
            self.take_screenshot(screenshot_path)

        self.transform.disable()

    def post_render(self) -> None:
        self.window.flip()
        self.onetime_geoms.clear()

    def clear(self):
        self.geoms.clear()
        self.onetime_geoms.clear()

    def close(self):
        try:
            self.window.close()
        except:
            pass

    def last_frame(self) -> np.ndarray:
        # TODO Double flip necessary?
        # self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        # self.window.flip()
        frame = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 4))
        frame = frame[::-1, :, 0:3]
        return frame

    def take_screenshot(
        self,
        screenshot_path: Path,
    ) -> None:
        from PIL import Image
        frame = self.last_frame()
        im = Image.fromarray(frame)
        os.makedirs(screenshot_path, exist_ok=True)
        im.save(screenshot_path)

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

    def draw_circle(
        self,
        creator: str,
        origin: T_Position2D,
        radius: float,
        start_angle: T_Angle = 0,
        end_angle: T_Angle = TWO_PI,
        fill_color: Optional[Color] = None,
        border_color: Optional[Color] = None,
        line_width: float = 1.0,
        resolution: int = 30,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlCircle:
        circle = GlCircle(
            creator=creator,
            origin=origin,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            fill_color=fill_color,
            line_color=border_color,
            line_width=line_width,
            resolution=resolution
        )
        self.add_geom(geom=circle, is_persistent=is_persistent)
        return circle

    def draw_polygon(
        self,
        creator: str,
        vertices: T_Vertices,
        color: Color | list[Color],
        is_filled: bool = True,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlFilledPolygon | GlPolyLine:
        if is_filled:
            polygon = GlFilledPolygon(
                creator=creator,
                vertices=vertices,
                color=color
            )
        else:
            polygon = GlPolyLine(
                creator=creator,
                vertices=vertices,
                is_closed=False,
                color=color,
            )
        self.add_geom(geom=polygon, is_persistent=is_persistent)
        return polygon

    def draw_2d_shape(
        self,
        creator: str,
        vertices: T_Vertices,
        fill_color: Optional[Color] = None,
        border_color: Optional[Color] = None,
        translation: T_Position2D = (0.0, 0.0),
        rotation: float = 0.0,
        scale: T_Position2D = (1.0, 1.0),
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlFilledPolygon | GlPolyLine:
        if translation is not None or rotation is not None or scale is not None:
            poly_path = transform_vertices_2d(
                vertices=vertices,
                translation=translation,
                rotation=rotation,
                scale=scale
            )
        else:
            poly_path = vertices

        # Other draw functions add shape
        shape = None
        if fill_color is not None:
            shape = self.draw_polygon(
                creator=creator,
                vertices=poly_path,
                color=fill_color,
                is_filled=True,
                is_persistent=is_persistent,
                **kwargs
            )

        if border_color is not None:
            shape = self.draw_polyline(
                creator=creator,
                vertices=poly_path,
                is_closed=False,
                color=border_color,
                line_width=line_width,
                is_persistent=is_persistent
            )
        return shape

    def draw_polyline(
        self,
        creator: str,
        vertices: T_Vertices,
        is_closed: bool,
        color: Color | list[Color],
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlPolyLine:
        polyline = GlPolyLine(
            creator=creator,
            vertices=vertices,
            is_closed=is_closed,
            color=color,
            line_width=line_width
        )
        self.add_geom(geom=polyline, is_persistent=is_persistent)
        return polyline

    def draw_line(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        color: Color,
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlLine:
        gl_line = GlLine(
            creator=creator,
            start=start,
            end=end,
            color=color,
            line_width=line_width
        )
        self.add_geom(geom=gl_line, is_persistent=is_persistent)
        return gl_line

    def draw_dashed_line(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        color: Color,
        line_width: float = DEFAULT_LINE_WIDTH,
        spacing: float = 0.5,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlDashedLine:
        dashed_line = GlDashedLine(
            creator=creator,
            start=start,
            end=end,
            color=color,
            line_width=line_width,
            spacing=spacing,
        )
        self.add_geom(geom=dashed_line, is_persistent=is_persistent)
        return dashed_line

    def draw_2d_arrow(
        self,
        creator: str,
        origin: T_Position2D,
        angle: T_Angle,
        length: float,
        line_color: Color,
        arrow_head_color: Optional[Color] = None,
        line_width: float = DEFAULT_LINE_WIDTH,
        arrow_head_size: float = 0.5,
        arrow_head_offset: float = 1.0,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlArrow:
        arrow = GlArrow(
            creator=creator,
            origin=origin,
            angle=angle,
            length=length,
            line_color=line_color,
            arrow_head_color=arrow_head_color,
            line_width=line_width,
            arrow_head_size=arrow_head_size,
            arrow_head_offset=arrow_head_offset
        )
        self.add_geom(geom=arrow, is_persistent=is_persistent)
        return arrow

    def draw_label(
        self,
        creator: str,
        text: str,
        color: Color,
        font_name: str = None,
        font_size: float = 4.5,
        bold: bool = False,
        italic: bool = False,
        stretch: bool = False,
        x: int = 0,
        y: int = 0,
        width: float = None,
        height: float = None,
        anchor_x: str = 'center',
        anchor_y: str = 'center',
        align: str = 'left',
        multiline: bool = False,
        dpi: int = None,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> GlLabel:
        label = GlLabel(
            creator=creator,
            text=text,
            color=color,
            font_name=font_name,
            font_size=font_size,
            bold=bold,
            italic=italic,
            stretch=stretch,
            x=x,
            y=y,
            width=width,
            height=height,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            align=align,
            multiline=multiline,
            dpi=dpi
        )
        self.add_geom(geom=label, is_persistent=is_persistent)
        return label

    # ! Not part of BaseViewer API
    def draw_rgb_image(
        self,
        data: Union[np.ndarray, Tensor],
        pos: np.ndarray,
        scale: float = 1.0,
        **kwargs
    ) -> None:
        from pyglet.gl.gl import GLubyte
        data = data.numpy() if isinstance(data, Tensor) else data
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

    def __del__(self):
        self.close()
