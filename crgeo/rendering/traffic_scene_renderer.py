import logging
import os
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any, Optional, Dict, Union, Sequence, Tuple, overload
from typing_extensions import Literal

# Checks if rendering is in rendering headless mode, can be activated by 'export PYGLET_HEADLESS=...' in environment
import pyglet
from commonroad.scenario.trajectory import State
from crgeo.common.io_extensions.obstacle import get_state_list
from crgeo.common.utils.datetime import get_timestamp_filename
if "PYGLET_HEADLESS" in os.environ:
    pyglet.options["headless"] = True

from pyglet import gl
import numpy as np
from crgeo.rendering.viewer.viewer_2d import Viewer2D
from crgeo.common.geometry.helpers import princip
from commonroad.scenario.scenario import Scenario
from crgeo.common.geometry.continuous_polyline import ContinuousPolyline
from crgeo.common.utils.string import numpy_prettify
from crgeo.rendering.defaults import DEFAULT_FPS, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, ColorTheme
from crgeo.rendering.plugins.defaults import DEFAULT_RENDERING_PLUGINS
from crgeo.rendering.types import RenderParams, SkipRenderInterrupt, T_RendererPlugin, T_Frame
from crgeo.rendering.color_utils import T_ColorTuple3

if TYPE_CHECKING:
    from pyglet.window import key

logger = logging.getLogger(__name__)


class RenderFailureException(RuntimeError):
    pass

@dataclass
class TrafficSceneRendererOptions:
    """
    Configiration options for TrafficSceneRenderer.
    """
    
    cache_last: bool = True
    camera_auto_rotation: bool = False
    camera_init_rotation: Optional[float] = None
    camera_rotation_speed: float = 0.7
    camera_init_strategy: Literal["map", "traffic"] = "map"
    caption: str = "CommonRoad-Geometric Traffic Scene Renderer"
    disable_overlays: bool = False
    export_dir: Optional[str] = None
    camera_follow: bool = False
    fps: Union[int, float] = DEFAULT_FPS
    minimize_window: bool = False
    plugins: Optional[Sequence[T_RendererPlugin]] = None
    pretty: bool = True
    render_freq: int = 1
    skip_redundant: bool = True
    theme: ColorTheme = ColorTheme.BRIGHT
    view_range: Optional[float] = None
    resizable: bool = False
    window_height: int = DEFAULT_WINDOW_HEIGHT
    window_size_multiplier: Optional[float] = 1.0
    window_width: int = DEFAULT_WINDOW_WIDTH
    swallow_skip_interrupts: bool = False
    sleep_when_inactive: bool = False
    smoothing: bool = True
    transparent_screenshots: bool = False
    disabled: bool = False
    override_background_color: Optional[T_ColorTuple3] = None



def jit_viewer(func):
    def _decorator(self, *args, **kwargs):
        if self._viewer is None:
            self.start()
        return func(self, *args, **kwargs)
    return _decorator


class TrafficSceneRenderer:
    """
    TrafficSceneRenderer is a fully customizable OpenGL-based renderer allowing
    users to visualize traffic scenarios, be it in real time or via recorded videos.
    """

    def __init__(
        self,
        options: Optional[Union[Dict[str, Any], TrafficSceneRendererOptions]] = None
    ) -> None:
        if options is None:
            options = TrafficSceneRendererOptions()
        elif not isinstance(options, TrafficSceneRendererOptions):
            options = TrafficSceneRendererOptions(**options)
        self.options = options
        self._plugins = options.plugins or DEFAULT_RENDERING_PLUGINS
        self._size_multiplier = 1.0 if self.options.window_size_multiplier is None else self.options.window_size_multiplier
        self._window_w = int(options.window_width*self._size_multiplier)
        self._window_h = int(options.window_height*self._size_multiplier)
        # TODO: instead just save options
        if options.camera_follow:
            self._view_range_input = options.view_range if options.view_range is not None else 150.0
        else:
            self._view_range_input = options.view_range
        self._fps = options.fps
        self._export_dir = options.export_dir
        self._frame_period = 1 / options.fps if options.fps > 0 else 0.0
        self._render_freq = options.render_freq
        self._last_render_time: Optional[float] = None
        self._current_render_timestep: Optional[int] = None
        self._last_render_timestep: Optional[int] = None
        self._last_frame: Optional[T_Frame] = None
        self._rotation: Optional[float] = 0.0
        self._disable_overlays = options.disable_overlays
        self._scenario: Optional[Scenario] = None
        self._call_count = -1
        self._suppress_warning = False
        self._viewer: Optional[Viewer2D] = None

    @property
    def keys(self) -> 'key.KeyStateHandler':
        if self._viewer is None:
            self.start()
        return self._viewer.keys

    @property
    def scenario(self) -> Scenario:
        if self._scenario is None:
            raise ValueError("scenario attribute for TrafficSceneRenderer has not been set")
        return self._scenario

    @scenario.setter
    @jit_viewer
    def scenario(self, scenario: Scenario) -> None:
        # Remove this if we ever start modifying the lanelet network of a scenario
        is_new_scenario = self._scenario is None or scenario.scenario_id != self._scenario.scenario_id
        respawn_camera = self.options.camera_follow and (self._last_render_timestep is not None and self._last_render_timestep > self._current_render_timestep)
        if not is_new_scenario and not respawn_camera:
            return

        if type(scenario) is tuple:
            scenario = scenario[0]
        self._scenario = scenario
        lanelet_network = scenario.lanelet_network

        if self.options.camera_init_strategy == "map":
            self._min_x = min([ll.center_vertices[:, 0].min() for ll in lanelet_network.lanelets])
            self._max_x = max([ll.center_vertices[:, 0].max() for ll in lanelet_network.lanelets])
            self._min_y = min([ll.center_vertices[:, 1].min() for ll in lanelet_network.lanelets])
            self._max_y = max([ll.center_vertices[:, 1].max() for ll in lanelet_network.lanelets])
        elif self.options.camera_init_strategy == "traffic":
            self._min_x = min([min([s.position[0] for s in get_state_list(o)]) for o in scenario.dynamic_obstacles])
            self._max_x = max([max([s.position[0] for s in get_state_list(o)]) for o in scenario.dynamic_obstacles])
            self._min_y = min([min([s.position[1] for s in get_state_list(o)]) for o in scenario.dynamic_obstacles])
            self._max_y = max([max([s.position[1] for s in get_state_list(o)]) for o in scenario.dynamic_obstacles])
        else:
            raise ValueError(self.options.camera_init_strategy)
        
        dx = self._max_x - self._min_x
        dy = self._max_y - self._min_y
        self._center_x = (self._min_x + self._max_x) / 2
        self._center_y = (self._min_y + self._max_y) / 2
        dmax = max(dx, dy) * 1.15

        self._lanelet_network = lanelet_network
        self._lanelet_paths = {ll.lanelet_id: ContinuousPolyline(ll.center_vertices) for ll in lanelet_network.lanelets}
        
        if self.options.camera_init_rotation is not None:
            self._rotation = self.options.camera_init_rotation
        else:
            # setting default rotation to align with road
            distances = {lid: path.get_projected_distance(np.array([self._center_x, self._center_y])) for lid, path in self._lanelet_paths.items()}
            closest_lanelet_id = sorted(distances.keys(), key=lambda x: distances[x])[0]
            closest_lanelet_orientation = self._lanelet_paths[closest_lanelet_id].get_projected_direction(np.array([self._center_x, self._center_y]))
            self._rotation = -closest_lanelet_orientation

        self._view_range = dmax if self._view_range_input is None else self._view_range_input
        self._viewer.clear_geoms()

    @property
    def size(self) -> Tuple[int, int]:
        return self._viewer.size

    def start(self) -> None:
        if self._viewer is None:
            from crgeo.rendering.viewer.viewer_2d import Viewer2D
            self._viewer = Viewer2D(
                self._window_w,
                self._window_h,
                linewidth_multiplier=1.0 + self._size_multiplier*4.5, # TODO
                minimize=self.options.minimize_window,
                theme=self.options.theme,
                caption=self.options.caption,
                smoothing=self.options.smoothing,
                pretty=self.options.pretty,
                override_background_color=self.options.override_background_color,
                resizable=self.options.resizable,
                transparent_screenshots=self.options.transparent_screenshots
            )

    def close(self) -> None:
        if self._viewer is not None and self._viewer.success:
            self._viewer.window.close()

    @jit_viewer
    def _set_view(
        self, 
        position: np.ndarray,
        orientation: float,
        range: float
    ):
        self._center_x = position[0]
        self._center_y = position[1]
        self._viewer.transform.rotation = np.pi/2 - orientation
        self._view_range = range
        self._view_range_input = range

    @overload
    def render(
        self,
        render_params: RenderParams,
        return_rgb_array: Literal[False]
    ) -> None:
        ...

    @overload
    def render(
        self,
        render_params: RenderParams
    ) -> None:
        ...

    @overload
    def render(
        self,
        render_params: RenderParams,
        return_rgb_array: Literal[True]
    ) -> np.ndarray:
        ...

    @jit_viewer
    def render(
        self,
        *,
        render_params: RenderParams,
        return_rgb_array: bool = False
    ) -> T_Frame:
        self._call_count += 1
        self._current_render_timestep = render_params.time_step

        assert self._viewer is not None

        if self.options.disabled:
            return self._last_frame

        if not self._viewer.pre_render():
            return self._last_frame

        if self._call_count % self._render_freq != 0:
            return self._last_frame

        if self.options.sleep_when_inactive and not self._viewer.window._active:
            return self._last_frame

        if self.options.skip_redundant and self._scenario is not None and (render_params.scenario is None or render_params.scenario.scenario_id == self._scenario.scenario_id) \
             and render_params.time_step == self._last_render_timestep:
            if not self._suppress_warning:
                logger.warning(f"Called render twice for same time-step {render_params.time_step=}. Ignoring call and returning cached image.")
                self._suppress_warning = True
            if self._last_frame is not None:
                return self._last_frame

        if render_params.render_kwargs is None:
            render_params.render_kwargs = {}

        overlays = render_params.render_kwargs.get('overlays', None)

        if not self._viewer.success:
            raise RenderFailureException("Trying to render without viewer successfully initialized")

        if render_params.scenario is not None:
            self.scenario = render_params.scenario
        else:
            render_params.scenario = self._scenario

        if self._scenario is None:
            raise RenderFailureException("Scenario has not been set")

        if self.options.camera_follow:
            follow_state: Optional[State] = None
            if render_params.ego_vehicle is not None:
                follow_state = render_params.ego_vehicle.state
            elif render_params.simulation.current_obstacles:
                try:
                    follow_state = render_params.simulation.current_obstacles[self._viewer.focus_obstacle_idx].state_at_time(
                        render_params.time_step
                    )
                except IndexError:
                    follow_state = render_params.simulation.current_obstacles[0].state_at_time(
                        render_params.time_step
                    )
            if follow_state is not None:
                self._center_x = follow_state.position[0]
                self._center_y = follow_state.position[1]
                if self.options.camera_auto_rotation:
                    if self._rotation is None:
                        self._rotation = follow_state.orientation
                    else:
                        self._rotation += render_params.simulation.dt * self.options.camera_rotation_speed * princip(
                            -follow_state.orientation + np.pi / 2.0 - self._rotation
                        )
        self._viewer.transform.rotation = self._rotation
        
        if render_params.camera_view is not None:
            self._set_view(
                position=render_params.camera_view.position,
                orientation=render_params.camera_view.orientation,
                range=render_params.camera_view.range
            )

        min_x, max_x, min_y, max_y = self._viewer.fit_into_viewport(
            self._center_x - self._view_range / 2,
            self._center_x + self._view_range / 2,
            self._center_y - self._view_range / 2,
            self._center_y + self._view_range / 2
        )
        self._viewer.set_bounds(min_x, max_x, min_y, max_y)

        self._viewer.transform.rotation

        def render_objects():
            t = self._viewer.transform
            t.enable()

            if render_params.render_kwargs is None:
                render_params.render_kwargs = {}
            render_params.render_kwargs['plugins'] = self._plugins # hack for giving plugins access to each other
            for renderer_plugin in self._plugins:
                try:
                    renderer_plugin(self._viewer, render_params)
                except SkipRenderInterrupt:
                    if not self.options.swallow_skip_interrupts:
                        raise

            self._viewer.render()
            t.disable()

            if not self._disable_overlays:
                if self._viewer.overlay_labels is not None:
                    for label in self._viewer.overlay_labels:
                        label.render1()
                self._viewer.overlay_labels = []
                self._render_overlays(overlays)

        win = self._viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()

        gl.glViewport(0, 0, self._viewer.width, self._viewer.height)

        render_objects()

        arr: T_Frame = None
        if return_rgb_array:
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(self._viewer.height, self._viewer.width, 4)
            arr = arr[::-1, :, 0:3]

            if self._export_dir is not None:
                # TODO let viewer handle
                from PIL import Image
                output_dir = os.path.join(self._export_dir)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'capture_{self._viewer.width}x{self._viewer.height}_{len(self._plugins)}_plugins_{self._scenario.scenario_id}_{render_params.time_step}_{hash(self._viewer)}.png')
                im = Image.fromarray(arr)
                im.save(output_file)

        win.flip()

        self._viewer.post_render()

        now = time.time()
        if self._last_render_time is not None:
            dt = now - self._last_render_time
            if dt < self._frame_period:
                time.sleep(self._frame_period - dt)
        self._last_render_time = now

        self._last_render_timestep = render_params.time_step
        if self.options.cache_last:
            self._last_frame = arr
        return arr

    @jit_viewer
    def screenshot(
        self,
        output_file: str,
        queued: bool = True,
        size: Optional[Tuple[int, int]] = None
    ) -> None: 
        self._viewer.screenshot(
            output_file=output_file,
            queued=queued,
            size=size
        )
        if queued:
            logger.info(f"Queued screenshot - will be exported to '{output_file}'")
        else:
            logger.info(f"Captured screenshot - exported to '{output_file}'")

    def _render_overlays(self, overlays: Optional[Dict[str, Union[str, float, np.ndarray]]]) -> None:
        if overlays is None:
            return

        # TODO: Make less hacky
        text = '\n'.join([f"{(k + ':' if k else ''):<35}{numpy_prettify(v):<30}" for k, v in overlays.items()])
        self._viewer.overlay_label.text = text
        self._viewer.overlay_label.draw()

    def __repr__(self) -> str:
        return f"TrafficSceneRenderer({len(self._plugins)} plugins)"
