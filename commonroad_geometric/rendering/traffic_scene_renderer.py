import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pyglet
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.base_renderer_plugin import T_RendererPlugin
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import T_CameraPlugin
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.plugins.cameras.follow_vehicle_camera import FollowVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.defaults import DEFAULT_RENDERING_PLUGINS
from commonroad_geometric.rendering.plugins.implementations.render_overlays_plugin import RenderOverlayPlugin
from commonroad_geometric.rendering.types import RenderParams, SkipRenderInterrupt, T_Frame
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer, T_ViewerOptions
# from commonroad_geometric.rendering.viewer.open3d.open3d_viewer import Open3DViewer, Open3DViewerOptions
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D, GLViewerOptions

# Checks if rendering is in rendering headless mode, can be activated by 'export PYGLET_HEADLESS=...' in environment
if "PYGLET_HEADLESS" in os.environ:
    pyglet.options["headless"] = True

logger = logging.getLogger(__name__)


class RenderFailureException(RuntimeError):
    pass


DEFAULT_FPS = 40


@dataclass
class FrameInfo:
    time: float
    timestep: int
    scenario: Optional[Scenario]
    frame: Optional[T_Frame]


@dataclass
class TrafficSceneRendererOptions:
    """
    Configuration options for TrafficSceneRenderer
    """
    # Basic options
    viewer_options: T_ViewerOptions = field(default_factory=GLViewerOptions)
    camera: T_CameraPlugin = field(default_factory=GlobalMapCamera)
    plugins: Sequence[T_RendererPlugin] = field(default_factory=lambda: DEFAULT_RENDERING_PLUGINS)
    fps: Union[int, float] = DEFAULT_FPS
    theme: ColorTheme = ColorTheme.BRIGHT
    overlay: RenderOverlayPlugin = field(default_factory=RenderOverlayPlugin)
    disable_overlays: bool = False
    export_dir: Optional[Path] = None
    cache_last: bool = True
    # Advanced options for (partially) disabling rendering
    is_disabled: bool = False
    render_freq: int = 1
    skip_redundant_renders: bool = True
    sleep_when_inactive: bool = False
    swallow_skip_interrupts: bool = False

    @property
    def frame_period(self) -> float:
        return 1 / self.fps if self.fps > 0 else 0.0


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
        if isinstance(options.viewer_options, dict):
            options.viewer_options = GLViewerOptions(**options.viewer_options)
        if options.camera == {}:
            options.camera = EgoVehicleCamera(view_range=200.0, camera_rotation_speed=0.0)
        self.options = options
        self._viewer: Optional[BaseViewer] = None

        self._camera = options.camera
        self._plugins = options.plugins or DEFAULT_RENDERING_PLUGINS
        self._last_frame_info: Optional[FrameInfo] = None

        self._call_count = -1
        self._skip_next_frames: int = 0
        self._frame_count: int = 0

        self._suppress_next_warning = False

    @property
    def viewer(self) -> BaseViewer:
        return self._viewer

    @property
    def size(self) -> Tuple[int, int]:
        return self.viewer.width, self.viewer.height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self) -> None:
        if self._viewer is None:
            match self.options.viewer_options:
                case GLViewerOptions() as gl_viewer_options:
                    self._viewer = GLViewer2D(options=gl_viewer_options)
                case Open3DViewerOptions() as open3d_options:
                    self._viewer = Open3DViewer(options=open3d_options)
                case _:
                    raise AttributeError(f"The passed viewer options are not supported: {self.options.viewer_options},"
                                         f"required: Open3DViewerOptions or GLViewerOptions")

    def close(self) -> None:
        if self._viewer is not None and self._viewer.is_initialized:
            self._viewer.close()

    def disable(self) -> None:
        self.options.is_disabled = True

    def enable(self) -> None:
        self.options.is_disabled = False

    def skip_frames(self, n: int = 1) -> None:
        self._skip_next_frames = n

    @jit_viewer
    def render(
        self,
        *,
        render_params: RenderParams,
        return_frame: bool = False
    ) -> Optional[T_Frame]:
        self._call_count += 1

        assert self._viewer is not None
        assert render_params is not None

        if any((
            self.options.is_disabled,
            self._skip_next_frames > 0,
            self._call_count % self.options.render_freq != 0,
            self.options.sleep_when_inactive and not self._viewer.is_active,
            self._viewer.is_minimized,
        )):
            self._skip_next_frames -= 1
            if self._last_frame_info:
                return self._last_frame_info.frame
            else:
                return None

        if self._last_frame_info is not None and all((
            self.options.skip_redundant_renders,
            self._last_frame_info is not None,
            render_params.time_step == self._last_frame_info.timestep,
            render_params.scenario.scenario_id == self._last_frame_info.scenario.scenario_id
        )):
            if not self._suppress_next_warning:
                logger.warning(
                    f"Called render twice for same time-step {render_params.time_step=}. Ignoring call and returning cached image.")
                self._suppress_next_warning = True
            return self._last_frame_info.frame

        if not self._viewer.is_initialized:
            raise RenderFailureException("Trying to render with unsuccessfully initialized viewer")

        self._camera(self._viewer, render_params)
        self._viewer.pre_render()

        # render_params.render_kwargs['plugins'] = self._plugins  # hack for giving plugins access to each other
        
        for renderer_plugin in self._plugins:
            try:
                renderer_plugin(self._viewer, render_params)
            except SkipRenderInterrupt as e:
                if not self.options.swallow_skip_interrupts:
                    raise e

        screenshot_path = None
        if self.options.export_dir is not None:
            filename = f'capture_{self._viewer.width}x{self._viewer.height}_{len(self._plugins)}_plugins_{self._last_frame_info.scenario.scenario_id}_{render_params.time_step}_{hash(self._viewer)}.png'
            screenshot_path = self.options.export_dir / filename

        self._viewer.render(screenshot_path)

        frame: T_Frame = None
        if return_frame:
            frame = self._viewer.last_frame()

        self._viewer.post_render()

        now = time.time()
        if self._last_frame_info is not None:
            dt = now - self._last_frame_info.time
            if dt < self.options.frame_period:
                time.sleep(self.options.frame_period - dt)

        # frame_diff = np.abs(self._last_frame_info.frame - frame).sum() if self._last_frame_info is not None else None
        # print(f"Rendered {render_params.scenario.scenario_id} ({render_params.time_step}) | {frame_diff=}")

        self._last_frame_info = FrameInfo(
            time=now,
            timestep=render_params.time_step,
            scenario=render_params.scenario,
            frame=frame
        )

        return frame

    @jit_viewer
    def screenshot(
        self,
        export_path: Path,
        queued: bool = True,
        size: Optional[Tuple[int, int]] = None
    ) -> None:
        self._viewer.take_screenshot(
            export_path=export_path,
            queued=queued,
            size=size
        )
        if queued:
            logger.info(f"Queued screenshot - will be exported to '{export_path}'")
        else:
            logger.info(f"Captured screenshot - exported to '{export_path}'")

    def _render_overlays(self, overlays: Optional[Dict[str, Union[str, float, np.ndarray]]]) -> None:
        if overlays is None:
            return

        # TODO: Make less hacky
        text = '\n'.join([f"{(k + ':' if k else ''):<35}{numpy_prettify(v):<30}" for k, v in overlays.items()])
        self._viewer.overlay_label.text = text
        self._viewer.overlay_label.draw()

    def __repr__(self) -> str:
        return f"TrafficSceneRenderer({len(self._plugins)} plugins)"

    def __getstate__(self):
        # When pickling, return None instead of the object's state
        return None

    def __setstate__(self, state):
        pass