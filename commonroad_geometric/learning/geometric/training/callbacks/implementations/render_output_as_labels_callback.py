import os
from typing import Callable

from torch import Tensor
from pathlib import Path
from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, StepCallbackParams
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions


class RenderOutputAsLabels(BaseCallback[StepCallbackParams]):
    def __init__(
        self,
        scenario_dir: Path,
        rendering_frequency: int = 1000,
        render_original_parameter: str = None,
        output_transform: Callable[[Tensor], Tensor] = lambda params: params,
        original_parameter_transform: Callable[[Tensor], Tensor] = lambda params: params,
        save_directory: Path = None,
    ):
        super().__init__()
        self._scenario_dir = scenario_dir
        self._rendering_frequency = rendering_frequency
        self._render_original_parameter = render_original_parameter
        self._output_transform = output_transform
        self._original_parameter_transform = original_parameter_transform
        self._save_directory = save_directory

    def __call__(
        self,
        params: StepCallbackParams,
    ):
        if params.ctx.epoch % self._rendering_frequency == 0:
            _renderer = TrafficSceneRenderer(
                options=TrafficSceneRendererOptions(
                    viewer_options=GLViewerOptions(
                        window_height=1080,
                        window_width=1920,
                    ),
                    plugins=[
                        RenderObstaclePlugin(
                            from_graph=True,
                        ),
                        RenderLaneletNetworkPlugin(
                            from_graph=True
                        )
                    ],
                ),
            )

            # Apply optional transformation to the output
            params.output = self._output_transform(params.output)

            # Get only the first minibatch
            batch = next(iter(params.batch.cpu()))
            output = params.output.cpu()[:batch.vertices[self._render_original_parameter].shape[0]]

            # TODO Fix implementation of data-only rendering
            frame = _renderer.render(
                render_params=RenderParams(
                    data=params.batch,
                    time_step=0,
                    render_kwargs={'labels': output.tolist()}
                ),
                return_frame=True
            )

            if self._save_directory is not None:
                from PIL import Image
                output_file = self._save_directory.joinpath(f'epoch_{params.ctx.epoch}_recreated_img.png')
                im = Image.fromarray(frame)
                if not os.path.exists(self._save_directory):
                    self._save_directory.mkdir(parents=True, exist_ok=True)
                im.save(output_file)

            if self._render_original_parameter:
                labels = self._original_parameter_transform(batch.v[self._render_original_parameter]).tolist()
                frame = _renderer.render(
                    render_params=RenderParams(
                        data=batch.cpu(),
                        time_step=0,
                        render_kwargs={
                            'labels': labels,
                            'return_frames': True
                        }
                    )
                )
                if self._save_directory is not None:
                    from PIL import Image
                    output_file = self._save_directory.joinpath(f'epoch_{params.ctx.epoch}_original_img.png')
                    im = Image.fromarray(frame)
                    if not os.path.exists(self._save_directory):
                        os.makedirs(self._save_directory)
                    im.save(output_file)
            _renderer.close()
