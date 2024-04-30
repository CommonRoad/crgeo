"""This module contains the abstract base class of for GLViewer2D and Open3DViewer"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.common.geometry.helpers import TWO_PI
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.types import CameraView2D, T_Angle, T_Position2D, T_Vertices
from commonroad_geometric.rendering.viewer.geoms.arrow import Arrow
from commonroad_geometric.rendering.viewer.geoms.base_geom import BaseGeom
from commonroad_geometric.rendering.viewer.geoms.circle import InterpolatedCircle
from commonroad_geometric.rendering.viewer.geoms.polygon import FilledPolygon
from commonroad_geometric.rendering.viewer.geoms.polyline import PolyLine
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH

DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 800

T_Viewer = TypeVar("T_Viewer", bound="BaseViewer")
T_ViewerOptions = TypeVar("T_ViewerOptions", bound="ViewerOptions")
T_BaseGeom = TypeVar("T_BaseGeom", bound="BaseGeom")


@dataclass
class ViewerOptions(ABC):
    caption: str = "CommonRoad-Geometric Traffic Scene Renderer"
    window_width: int = DEFAULT_WINDOW_WIDTH
    window_height: int = DEFAULT_WINDOW_HEIGHT
    window_scaling_factor: float = 1.0  # Set to 2.0 for 4K screen
    minimize_window: bool = False
    theme: ColorTheme = ColorTheme.BRIGHT

    @property
    def scaled_window_width(self) -> int:
        scaling_factor = self.window_scaling_factor if self.window_scaling_factor is not None else 1.0
        return int(self.window_width * scaling_factor)

    @property
    def scaled_window_height(self) -> int:
        scaling_factor = self.window_scaling_factor if self.window_scaling_factor is not None else 1.0
        return int(self.window_height * scaling_factor)


class BaseViewer(ABC, Generic[T_ViewerOptions, T_BaseGeom]):
    """
    This abstract class defines the interface methods for the rendering framework that shall be used.
    """

    def __init__(
        self,
        options: T_ViewerOptions,
    ) -> None:
        self.is_initialized = False
        self._options = options
        self._xlim: tuple[float, float] = (0, 0)
        self._ylim: tuple[float, float] = (0, 0)
        # Scene attributes:
        self.geoms: list[T_BaseGeom] = []
        self.onetime_geoms: list[T_BaseGeom] = []
        # For shared randomness between plugins
        rng = np.random.default_rng(seed=0)
        self.shared_rng_cache = CachedRNG(rng=rng.random)

    @property
    def options(self) -> T_ViewerOptions:
        return self._options

    def add_geom(
        self,
        geom: BaseGeom,
        is_persistent: bool = False
    ):
        r"""
        Adds a geometry object to the viewer.

        Args:
            geom (BaseGeom): The geometry object.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
        """
        if is_persistent:
            self.geoms.append(geom)
        else:
            self.onetime_geoms.append(geom)

    def clear_geoms(self) -> None:
        self.geoms = []

    @property
    @abstractmethod
    def is_active(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_minimized(self) -> bool:
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...

    @property
    @abstractmethod
    def xlim(self) -> tuple[float, float]:
        ...

    @property
    @abstractmethod
    def ylim(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def set_view(self, camera_view: CameraView2D):
        ...

    @abstractmethod
    def pre_render(self):
        ...

    @abstractmethod
    def render(
        self,
        screenshot_path: Optional[Path] = None
    ):
        ...

    @abstractmethod
    def post_render(self):
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def last_frame(self) -> np.ndarray:
        ...

    @abstractmethod
    def take_screenshot(
        self,
        export_path: Path
    ):
        ...

    # Low-level drawing functions
    @abstractmethod
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
    ) -> InterpolatedCircle:
        r"""
        Draws a circle around the origin with given radius with:

        Args:
            creator (str): Creator of the circle, e.g. __name__ of the render plugin which called draw_circle.
            origin (T_Position2D): Origin/center of the circle.
            radius (float): Radius of the circle.
            start_angle (T_Angle): Starting angle to draw the perimeter point.
            end_angle (T_Angle): End angle to draw the perimeter point.
            fill_color (Optional[Color]): Optional color for the inside of the circle. Defaults to None (no color).
            border_color (Optional[Color]): Optional color for the border of the circle. Defaults to None (no color).
            line_width (float): Width of the border line. Defaults to 1.0.
            resolution (int): Resolution of the circle - number of points representing perimeter of the circle.
                              Defaults to 30.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The circle
        """
        ...

    @abstractmethod
    def draw_polygon(
        self,
        creator: str,
        vertices: T_Vertices,
        color: Color | list[Color],
        is_filled: bool = True,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> FilledPolygon | PolyLine:
        r"""
        Draws a polygon with:

        Args:
            creator (str): Creator of the polygon, e.g. __name__ of the render plugin which called draw_polygon.
            vertices (T_Vertices): 2D or 3D array of vertices of the polygon.
            color (Color | list[Color]): Either one color which is applied to the whole polygon or a list of colors with
                                         as many elements as there are vertices, where each color is applied to the
                                         respective vertex in the order given by the list.
            is_filled (bool): If True, the inside of the polygon will be filled. Defaults to True.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The polygon
        """
        ...

    @abstractmethod
    def draw_2d_shape(
        self,
        creator: str,
        vertices: T_Vertices,
        fill_color: Optional[Color] = None,
        border_color: Optional[Color | list[Color]] = None,
        translation: T_Position2D = (0.0, 0.0),
        rotation: float = 0.0,
        scale: T_Position2D = (1.0, 1.0),
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> FilledPolygon | PolyLine:
        r"""
        Draws an arbitrary 2D shape with:

        Args:
            creator (str): Creator of the polygon, e.g. __name__ of the render plugin which called draw_polygon.
            vertices (T_Vertices): 2D or 3D array of vertices of the shape.
            fill_color (Optional[Color]): Optional color for the inside of the shape. Defaults to None (no color).
            border_color (Optional[Color | list[Color]]): Optional color for the border of the shape. Either one color
                                                          which is applied to the whole border or a list of colors with
                                                          as many elements as there are vertices, where each color is
                                                          applied to the respective vertex in the order given by the
                                                          list. Defaults to None (no color).
            translation (Optional[T_Position2D]): Optional position shift (translation) applied to each vertex.
            rotation (Optional[T_Angle]): Optional angle shift (rotation) applied to each vertex.
            scale (Optional[T_Position2D]): Optional scaling applied to each vertex.
            line_width (float): Width of the border line. Defaults to 1.0.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The shape
        """
        ...

    @abstractmethod
    def draw_polyline(
        self,
        creator: str,
        vertices: T_Vertices,
        is_closed: bool,
        color: Color | list[Color],
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> PolyLine:
        r"""
        Draws a polyline consisting of vertices that are connected to each other in the order given in the array with:

        Args:
            creator (str): Creator of the polyline, e.g. __name__ of the render plugin which called draw_polyline.
            vertices (T_Vertices): 2D or 3D array of vertices.
            is_closed (bool): If True, the last and first point are connected to form a closed polyline.
            color (Color | list[Color]): Either one color which is applied to all line segments or a list of colors with
                                         as many elements as there are line segments, where each color is applied to the
                                         respective segment in the order given by the list.
            line_width (float): The line width of the polyline.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The polyline
        """
        ...

    @abstractmethod
    def draw_line(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        color: Color,
        line_width: float = DEFAULT_LINE_WIDTH,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> PolyLine:
        r"""
        Draws a line from start to end with:

        Args:
            creator (str): Creator of the line, e.g. __name__ of the render plugin which called draw_line.
            start (T_Position2D): The 2D start point.
            end (T_Position2D): The 2D end point.
            color (Color): The color of the line.
            line_width (float): The width of the line.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The line
        """
        ...

    @abstractmethod
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
    ) -> PolyLine:
        r"""
        Draws a dashed line from start to end with:

        Args:
            creator (str): Creator of the dashed line, e.g. __name__ of the render plugin which called draw_dashed_line.
            start (T_Position2D): The 2D start point.
            end (T_Position2D): The 2D end point.
            color (Color): The color of the line.
            line_width (float): The line width of each line segment.
            spacing (float): The space between each line segment.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The dashed line
        """
        ...

    @abstractmethod
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
    ) -> Arrow:
        r"""
        Draws a 2D arrow from origin with angle and length, with:

        Args:
            creator (str): Creator of the arrow, e.g. __name__ of the render plugin which called draw_arrow.
            origin (T_Position2D): The origin of the arrow.
            angle (T_Angle): The angle of the arrow.
            length (float): The length of the arrow.
            line_color (Color): The color of the line part of the arrow.
            arrow_head_color (Color): Optional color of the arrow head. Defaults to line_color if None.
            line_width (float): The width of the line part of the arrow.
            arrow_head_size (float): Relative size of the arrow head in comparison to the line.
            arrow_head_offset (float): Offset of the arrow head from the line.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The 2D Arrow
        """
        ...

    def draw_arrow_from_to(
        self,
        creator: str,
        start: T_Position2D,
        end: T_Position2D,
        line_color: Color,
        arrow_head_color: Optional[Color] = None,
        line_width: float = DEFAULT_LINE_WIDTH,
        arrow_head_size: float = 0.5,
        arrow_head_offset: float = 1.0,
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> Arrow:
        r"""
        Draws a 2D arrow from start to end, with:

        Args:
            creator (str): Creator of the arrow, e.g. __name__ of the render plugin which called draw_arrow.
            start (T_Position2D): The 2D start point.
            end (T_Position2D): The 2D end point.
            line_color (Color): The color of the line part of the arrow.
            arrow_head_color (Color): Optional color of the arrow head. Defaults to line_color if None.
            line_width (float): The width of the line part of the arrow.
            arrow_head_size (float): Relative size of the arrow head in comparison to the line.
            arrow_head_offset (float): Offset of the arrow head from the line.
            is_persistent (bool): If True, the geometry object stays persistent across frames. Defaults to False.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            The 2D Arrow
        """
        length = np.linalg.norm(end - start)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        return self.draw_2d_arrow(
            creator=creator,
            origin=start,
            angle=angle,
            length=length,
            line_color=line_color,
            arrow_head_color=arrow_head_color,
            line_width=line_width,
            arrow_head_size=arrow_head_size,
            arrow_head_offset=arrow_head_offset,
            **kwargs
        )

    @abstractmethod
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
    ):
        ...
