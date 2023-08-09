import logging
import os
import time
from typing import Generic, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union, overload

import pyglet

if "PYGLET_HEADLESS" in os.environ:
    pyglet.options["headless"] = True

from pyglet import gl
import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.rendering.types import T_RendererPlugin, RenderParams
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D

log = logging.getLogger(__name__)

T_RenderParams = TypeVar("T_RenderParams", bound=RenderParams)

# TODO: delete?

class ScenarioGraphRendererOptions(NamedTuple):
    window_width: int = 1000
    window_height: int = 1000
    background: T_ColorTuple = (0.0, 0.0, 0.0, 1.0)
    graph_scenario_bounds_padding: float = 0.02
    plugins: Optional[Sequence[T_RendererPlugin]] = None
    left_vertices_attr: str = "left_vertices"
    right_vertices_attr: str = "right_vertices"


class Bounds(NamedTuple):
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class View(NamedTuple):
    position: Tuple[float, float]
    rotation: float
    view_radius: float


class ScenarioGraphRenderer(Generic[T_RenderParams]):
    # like TrafficSceneRenderer but without the need for a CommonRoad scenario

    def __init__(self, options: ScenarioGraphRendererOptions):
        self._options = options
        self._viewer = Viewer2D(options.window_width, options.window_height)

    def close(self) -> None:
        if self._viewer.success:
            self._viewer.window.close()

    def scenario_graph_bounds(
        self,
        data: Union[CommonRoadData, CommonRoadDataTemporal],
        padding: float = 0.0,
    ) -> Bounds:
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        for coords in data.lanelet[self._options.left_vertices_attr]:
            min_x = min(min_x, coords[:, 0].min().item())
            max_x = max(max_x, coords[:, 0].max().item())
            min_y = min(min_y, coords[:, 1].min().item())
            max_y = max(max_y, coords[:, 1].max().item())

        width = max_x - min_x
        height = max_y - min_y
        min_x -= width * padding
        max_x += width * padding
        min_y -= height * padding
        max_y += height * padding
        return Bounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

    def fit_into_viewport(self, bounds: Bounds) -> Bounds:
        min_x, max_x, min_y, max_y = bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height
        window_aspect_ratio = self._viewer.width / self._viewer.height
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
        return Bounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

    @overload
    def render(
        self,
        render_params: T_RenderParams,
        view: Optional[Union[View, Bounds]],
        overlay: Optional[str],
        return_rgb_array: Literal[False],
    ) -> None:
        ...

    @overload
    def render(
        self,
        render_params: T_RenderParams,
        view: Optional[Union[View, Bounds]],
        overlay: Optional[str],
        return_rgb_array: Literal[True],
    ) -> np.ndarray:
        ... # TODO: Type hints

    def render(
        self,
        render_params: T_RenderParams,
        view: Optional[Union[View, Bounds]] = None,
        overlay: Optional[str] = None,
        return_rgb_array: bool = False,
    ) -> Union[np.ndarray, None]:
        assert isinstance(render_params.data, (CommonRoadData, CommonRoadDataTemporal))
        window = self._viewer.window
        window.switch_to()
        gl.glClearColor(*self._options.background)
        window.clear()
        window.dispatch_events()

        # set view
        gl.glViewport(0, 0, self._viewer.width, self._viewer.height)
        if view is None:
            bounds = self.scenario_graph_bounds(
                render_params.data,
                padding=self._options.graph_scenario_bounds_padding,
            )
            bounds = self.fit_into_viewport(bounds)
            self._viewer.set_bounds(bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y)

        elif isinstance(view, View):
            bounds = self.fit_into_viewport(Bounds(
                min_x=view.position[0] - view.view_radius,
                max_x=view.position[0] + view.view_radius,
                min_y=view.position[1] - view.view_radius,
                max_y=view.position[1] + view.view_radius,
            ))
            self._viewer.set_bounds(bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y)
            self._viewer.transform.rotation = view.rotation

        elif isinstance(view, Bounds):
                self._viewer.set_bounds(view.min_x, view.max_x, view.min_y, view.max_y)

        self._viewer.transform.enable()

        if self._options.plugins is not None:
            for renderer_plugin in self._options.plugins:
                renderer_plugin(self._viewer, render_params)

        for geom in self._viewer.onetime_geoms:
            geom.render()
        self._viewer.onetime_geoms = []

        self._viewer.transform.disable()

        if overlay is None and render_params.time_step is not None:
            overlay = f"{render_params.time_step:04d}"
        if overlay is not None:
            self._viewer.overlay_label.text = overlay
            self._viewer.overlay_label.draw()

        img = None
        if return_rgb_array:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            img = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            img = img.reshape((self._viewer.height, self._viewer.width, 4))
            img = img[::-1, :, 0:3]

        window.flip()

        return img

    def render_sequence(
        self,
        data_list: List[CommonRoadData],
        render_params: T_RenderParams,
        delta_time: float,
        return_rgb_array: bool = False,
    ) -> Optional[List[np.ndarray]]:
        sequence = []
        last_frame = -1.0
        for data in data_list:
            now = time.perf_counter()
            wait_time = last_frame + delta_time - now
            if wait_time > 0.0:
                time.sleep(wait_time)
            last_frame = now

            render_params.data = data
            tic = time.perf_counter()
            img = self.render(render_params=render_params, return_rgb_array=return_rgb_array)
            log.debug(f"Render time: %.2fs", time.perf_counter() - tic)
            sequence.append(img)

        return sequence if return_rgb_array else None
