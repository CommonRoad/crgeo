import sys, os; sys.path.insert(0, os.getcwd())

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.color_gradient import ColorGradient
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.video_recording import save_video_from_frames
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

INPUT_SCENARIO = Path('data/other/ARG_Carcarana-1_7_T-1.xml')
MAX_TIMESTEPS = 300
VIDEO_OUTPUT_FILE = Path('tutorials/output/custom_rendering/video.gif')


@dataclass
class MyCustomRenderObstaclesPlugin(BaseRenderObstaclePlugin):
    gradient: Optional[ColorGradient] = None

    def render(self, viewer: T_Viewer, params: RenderParams) -> None:
        assert params.time_step is not None
        assert params.scenario is not None

        for obstacle in params.simulation.current_obstacles:
            state = state_at_time(obstacle, params.time_step)
            vertices = obstacle.obstacle_shape.vertices

            obstacle_color = self.obstacle_color
            if self.gradient is not None:
                obstacle_color = self.gradient[params.time_step]

            viewer.draw_2d_shape(
                creator=self.__class__.__name__,
                vertices=vertices,
                fill_color=Color((0, 0, 0)),
                border_color=obstacle_color,
                translation=state.position,
                rotation=-state.orientation,
                line_width=2,
            )


if __name__ == '__main__':
    renderer_plugins = [
        RenderLaneletNetworkPlugin(  # noqa: 405
            lanelet_linewidth=3.0,
        ),
        MyCustomRenderObstaclesPlugin(
            obstacle_color=Color('green'),  # Prioritizes gradient over obstacle_color
            gradient=ColorGradient(['green', 'red'], max_val=MAX_TIMESTEPS, min_val=0)
        ),
    ]

    renderer = TrafficSceneRenderer(
        options=TrafficSceneRendererOptions(
            # viewer_options=Open3DViewerOptions(),
            viewer_options=GLViewerOptions(  # change to Open3DViewerOptions to use Open3D backend
                window_width=1000,
                window_height=1000,
            ),
            plugins=renderer_plugins,
            fps=30
        )
    )
    video_frames: List[np.ndarray] = []

    simulation = ScenarioSimulation(initial_scenario=INPUT_SCENARIO)
    with simulation:
        for timestep, scenario in simulation(num_time_steps=MAX_TIMESTEPS):
            # Contains one frame for one renderer
            frames = simulation.render(
                renderers=[renderer],
                return_frames=True
            )

            video_frames.append(*frames)
    # print("Obstacle Color:{}".format(test_obstacle_color))
    print(f"Collected video frames: {len(video_frames)}")
    print(f"Saving video to {VIDEO_OUTPUT_FILE}")
    save_video_from_frames(frames=video_frames, output_file=VIDEO_OUTPUT_FILE)
