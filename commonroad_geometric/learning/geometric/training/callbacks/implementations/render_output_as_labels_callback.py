import os
from typing import Callable

from torch import Tensor

from commonroad_geometric.learning.geometric.training.callbacks.base_callback import BaseCallback, StepCallbackParams
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions


class RenderOutputAsLabels(BaseCallback[StepCallbackParams]):
    def __init__(
        self,
        scenario_dir: str,
        rendering_frequency: int = 1000,
        render_original_parameter: str = None,
        output_transform: Callable[[Tensor], Tensor] = lambda params: params,
        original_parameter_transform: Callable[[Tensor], Tensor] = lambda params: params,
        save_directory: str = None,
    ):
        self._scenario_dir = scenario_dir
        self._rendering_frequency = rendering_frequency
        self._render_original_parameter = render_original_parameter
        self._output_transform = output_transform
        self._original_parameter_transform = original_parameter_transform
        self._save_directory = save_directory
        pass

    def __call__(
        self,
        params: StepCallbackParams,
    ):

        if params.ctx.epoch % self._rendering_frequency == 0:
            from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
            from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin
            from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
            from commonroad_geometric.rendering.types import RenderParams
            from commonroad_geometric.dataset.iteration.timestep_iterator import TimeStepIterator
            _renderer = TrafficSceneRenderer(
                options=TrafficSceneRendererOptions(
                    window_height=1080,
                    window_width=1920,
                    plugins=[
                        RenderObstaclesPlugin(
                            from_graph=True,
                            font_size=5,
                            draw_index=True,
                        ),
                        RenderLaneletNetworkPlugin()
                    ],
                ),
            )
            _renderer.scenario = TimeStepIterator(f'{self._scenario_dir}/{params.kwargs["batch"]._global_store["scenario_id"][0]}.xml', loop=True).scenario,

            # Apply optional transformation to the output
            params.kwargs['output'] = self._output_transform(params.kwargs['output'])

            # Get only the first minibatch
            batch = next(iter(params.kwargs['batch'].cpu()))
            output = params.kwargs['output'].cpu()[:batch.v[self._render_original_parameter].shape[0]]

            frame = _renderer.render(
                return_rgb_array=True,
                render_params=RenderParams(
                    data=batch,
                    time_step=0,
                    render_kwargs={'labels': output.tolist()}
                )
            )

            if self._save_directory is not None:
                from PIL import Image
                output_file = os.path.join(self._save_directory, f'epoch_{params.ctx.epoch}_recreated_img.png')
                im = Image.fromarray(frame)
                if not os.path.exists(self._save_directory):
                    os.makedirs(self._save_directory)
                im.save(output_file)

            if self._render_original_parameter:
                frame = _renderer.render(
                    render_params=RenderParams(
                        data=batch.cpu(),
                        time_step=0,
                        render_kwargs={'labels': self._original_parameter_transform(batch.v[self._render_original_parameter]).tolist(), 'return_rgb_array': True}
                    )
                )
                if self._save_directory is not None:
                    from PIL import Image
                    output_file = os.path.join(self._save_directory, f'epoch_{params.ctx.epoch}_original_img.png')
                    im = Image.fromarray(frame)
                    if not os.path.exists(self._save_directory):
                        os.makedirs(self._save_directory)
                    im.save(output_file)
            _renderer.close()
