"""
    This module aims at implementing the functionality of the BaseViewer class with Open3D as its underlying render
    framework.
"""
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from open3d.cuda.pybind.geometry import AxisAlignedBoundingBox
from open3d.cuda.pybind.io import write_image
from open3d.cuda.pybind.visualization import webrtc_server
from open3d.cuda.pybind.visualization.gui import Application, Rect, SceneWidget, Window
from open3d.cuda.pybind.visualization.rendering import Open3DScene

from commonroad_geometric.common.geometry.helpers import TWO_PI
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.types import CameraView2D, T_Angle, T_Position2D, T_Vertices
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer, ViewerOptions
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_arrow import Open3DArrow2D, Open3DArrow3D
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_circle import Open3DCircle
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_dashed_line import Open3DDashedLine
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_geom import Open3DGeom
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polygon import Open3DFilledPolygon
from commonroad_geometric.rendering.viewer.open3d.geoms.open3d_polyline import Open3DPolyLine
from commonroad_geometric.rendering.viewer.utils import DEFAULT_LINE_WIDTH, transform_vertices_2d


@dataclass
class Open3DViewerOptions(ViewerOptions):
    enable_remote_render: bool = False  # if activated low_level rendering is activated by default


class Open3DViewer(BaseViewer[Open3DViewerOptions, Open3DGeom]):
    r"""
    Implementation of the BaseViewer interface using Open3D rendering functionality.
    """

    def __init__(
        self,
        options: Open3DViewerOptions
    ) -> None:
        super().__init__(options=options)
        if self.options.enable_remote_render:
            webrtc_server.enable_webrtc()
        self.instance = Application.instance
        self.instance.initialize()
        self.window: Window = Application.instance.create_window(
            title=self.options.caption,
            width=self.options.scaled_window_width,
            height=self.options.scaled_window_height
        )

        self.scene_widget: SceneWidget = self.setup_scene_widget()
        self.window.add_child(self.scene_widget)

        self.window.set_on_layout(self._on_layout)
        self.window.show(not self.options.minimize_window)

        self.is_initialized = True

    def setup_scene_widget(self) -> SceneWidget:
        scene_widget = SceneWidget()
        scene_widget.scene = Open3DScene(self.window.renderer)
        scene_widget.scene.show_axes(True)
        bbox = AxisAlignedBoundingBox([-10, -10, -10], [10, 10, 10])
        scene_widget.setup_camera(  # kwargs called arg0, arg1, etc. due to pybind
            60,  # field_of_view
            bbox,  # model_bounds
            [0, 0, 0]  # center_of_rotation
        )

        if self.options.theme == ColorTheme.BRIGHT:
            scene_widget.scene.set_background(np.asarray([1.0, 1.0, 1.0, 1.0]))
        else:
            scene_widget.scene.set_background(np.asarray([0.0, 0.0, 0.0, 1.0]))

        return scene_widget

    def _on_layout(self, layout_context):
        """
        Open3D specific callback getting called whenever the window is resized. Used by low level only.
        """
        r = self.window.content_rect
        self.scene_widget.frame = Rect(r.x, r.y, r.width, r.height)

    @property
    def is_active(self) -> bool:
        return self.window.is_active_window

    @property
    def is_minimized(self) -> bool:
        return not self.window.is_visible

    @property
    def width(self) -> int:
        return self.window.size.width

    @property
    def height(self) -> int:
        return self.window.size.height

    @property
    def xlim(self) -> Tuple[float, float]:
        return self._xlim

    @property
    def ylim(self) -> Tuple[float, float]:
        return self._ylim

    def set_view(self, camera_view: CameraView2D):
        center_x, center_y = camera_view.center_position
        center_z = 0

        back = center_x - camera_view.view_range / 2
        front = center_x + camera_view.view_range / 2
        left = center_y - camera_view.view_range / 2
        right = center_y + camera_view.view_range / 2
        bottom = center_z - camera_view.view_range / 2
        top = center_z + camera_view.view_range / 2

        assert front > back and right > left and top > bottom
        self._xlim = (back, front)
        self._ylim = (left, right)

        bounding_box = AxisAlignedBoundingBox([back, left, bottom], [front, right, top])
        center_3d = np.append(camera_view.center_position, 0)
        self.scene_widget.setup_camera(camera_view.view_range / 4, bounding_box, center_3d)

    def pre_render(self):
        pass

    def post_render(self):
        for geom in self.onetime_geoms:
            for index, geometry in enumerate(geom.o3d_geometries):
                uuid = f"{geom.uuid}-{index}"
                self.scene_widget.scene.remove_geometry(uuid)
        self.onetime_geoms.clear()

    def last_frame(self) -> np.ndarray:
        pass

    def take_screenshot(
        self,
        output_file: str,
        queued: bool = True
    ):
        # Note: only works when window is visible
        def on_image(image):
            quality = 9  # png
            if output_file.endswith(".jpg"):
                quality = 100
            write_image(
                filename=output_file,
                image=image,
                quality=quality
            )
        self.scene_widget.render_to_image(on_image)

    def render(
        self,
        screenshot_path: Optional[Path] = None
    ):
        for geom in chain(self.geoms, self.onetime_geoms):
            for index, (geometry, material_record) in enumerate(zip(geom.o3d_geometries, geom.o3d_material_records)):
                uuid = f"{geom.uuid}-{index}"
                if self.scene_widget.scene.has_geometry(uuid):
                    continue
                self.scene_widget.scene.add_geometry(
                    name=uuid,
                    geometry=geometry,
                    material=material_record
                )

        if screenshot_path is not None:
            self.take_screenshot(screenshot_path)

        self.instance.run_one_tick()
        self.window.post_redraw()

    def clear(self):
        for geom in chain(self.geoms, self.onetime_geoms):
            for index, _ in enumerate(zip(geom.o3d_geometries, geom.o3d_material_records)):
                uuid = f"{geom.uuid}-{index}"
                self.scene_widget.scene.remove_geometry(uuid)
        self.geoms.clear()
        self.onetime_geoms.clear()

    def close(self):
        self.window.close()

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
    ) -> Open3DCircle:
        circle = Open3DCircle(
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
    ) -> Open3DFilledPolygon | Open3DPolyLine:
        if is_filled:
            polygon = Open3DFilledPolygon(
                creator=creator,
                vertices=vertices,
                color=color
            )
        else:
            polygon = Open3DPolyLine(
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
    ) -> Open3DFilledPolygon | Open3DPolyLine:
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
    ) -> Open3DPolyLine:
        polyline = Open3DPolyLine(
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
        is_persistent: bool = False,
        **kwargs: dict[str, Any]
    ) -> Open3DPolyLine:
        vertices = np.array([start, end])
        return self.draw_polyline(
            creator=creator,
            vertices=vertices,
            is_closed=False,
            color=color,
            is_persistent=is_persistent,
            **kwargs
        )

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
    ) -> Open3DDashedLine:
        dashed_line = Open3DDashedLine(
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
        render_in_3d: bool = False,
        **kwargs: dict[str, Any]
    ) -> Open3DArrow2D | Open3DArrow3D:
        if render_in_3d:
            arrow = Open3DArrow3D(
                creator=creator,
                cylinder_radius=1,
                cone_radius=1,
                cylinder_height=length,
                cone_height=1,
                resolution=8,
                cylinder_split=4,
                cone_split=4
            )
        else:
            arrow = Open3DArrow2D(
                creator=creator,
                origin=origin,
                angle=angle,
                length=length,
                line_color=line_color,
                arrow_head_color=arrow_head_color,
                arrow_head_size=arrow_head_size,
                arrow_head_offset=arrow_head_offset
            )
        self.add_geom(geom=arrow, is_persistent=is_persistent)
        return arrow

    def draw_arrow_from_to(
        self,
        start: np.ndarray,
        end: np.ndarray,
        scale: float = 0.45,
        end_offset: float = 1.0,
        **kwargs
    ) -> Open3DArrow2D | Open3DArrow3D:
        # TODO
        pass

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
        # TODO
        pass
