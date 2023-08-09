import sys, os; sys.path.insert(0, os.getcwd())

from typing import List

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.logging import stdout
from commonroad_geometric.dataset.iteration import TimeStepIterator
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.plugins import *
from commonroad_geometric.rendering.traffic_scene_renderer import T_Frame, TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.video_recording import save_video_from_frames
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D

INPUT_SCENARIO = 'data/osm_recordings/DEU_Munich-1_114_0_time_steps_1000_V1_0.xml'
MAX_TIMESTEPS = 300
VIDEO_OUTPUT_FILE = 'tutorials/output/custom_rendering/video.gif'


class MyCustomRenderObstaclesPlugin(BaseRendererPlugin):  # noqa: 405
    def __init__(self) -> None:
        self._rng_cache = CachedRNG(np.random.random) # type: ignore

    def __call__(self, viewer: Viewer2D, params: RenderParams) -> None:  # noqa: 405
        assert params.time_step is not None
        assert params.scenario is not None
        for obstacle in params.simulation.current_obstacles:
            state = state_at_time(obstacle, params.time_step)
            vertices = obstacle.obstacle_shape.vertices # type: ignore
            obstacle_color = self._rng_cache(obstacle.obstacle_id, n=3) + (1.0,)
            viewer.draw_shape(
                vertices,
                state.position,
                state.orientation,
                color=obstacle_color,
                filled=True,
                linewidth=2,
                fill_color=(0, 0, 0),
                border=obstacle_color
            )


if __name__ == '__main__':
    timestep_iterator = TimeStepIterator(INPUT_SCENARIO, loop=True)
    renderer_plugins = [
        RenderLaneletNetworkPlugin(  # noqa: 405
            lanelet_linewidth=3.0,
            lanelet_color=(0.2, 0.2, 0.2)
            #lanelet_color=(0.0, 0.0, 1.0)
        ),
        #RenderObstaclesPlugin(),
        MyCustomRenderObstaclesPlugin(),
    ]
    renderer = TrafficSceneRenderer(
        options=TrafficSceneRendererOptions(
            window_height=1000,
            window_width=1000,
            plugins=renderer_plugins,
            fps=30
        )
    )
    video_frames: List[np.ndarray] = []
    for time_step in timestep_iterator:
        if len(video_frames) >= MAX_TIMESTEPS:
            break
        frame = renderer.render(
            return_rgb_array=True,
            render_params=RenderParams(  # noqa: 405
                scenario=timestep_iterator.scenario,
                time_step=time_step
            ),
        )
        video_frames.append(frame)
        stdout(f"Collected video frames: {len(video_frames)}")

    print(f"Saving video to {VIDEO_OUTPUT_FILE}")
    save_video_from_frames(frames=video_frames, output_file=VIDEO_OUTPUT_FILE)
