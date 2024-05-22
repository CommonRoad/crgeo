from dataclasses import dataclass
from typing import Optional

from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderObstacleLabels(BaseRenderObstaclePlugin):
    from_graph: bool = False
    graph_position_attr: Optional[str] = "pos"
    font_size: float = 7

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if self.from_graph:
            self._render_from_graph(viewer, params)
        else:
            self._render_from_scenario(viewer, params)

    def _render_from_scenario(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        raise NotImplementedError()

    def _render_from_graph(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return
        if params.render_kwargs is None or 'label_dict' not in params.render_kwargs:
            return

        label_dict = params.render_kwargs['label_dict']
        if not isinstance(label_dict, dict):
            return

        for index in range(params.data.v['num_nodes']):
            position = params.data.v[self.graph_position_attr][index].numpy()
            for dict_index, key in enumerate(label_dict):
                label = str(f"{key}:{label_dict[key][index].cpu().detach().numpy():.1f}")
                rng_cache = self.get_active_rng_cache(viewer, params)
                color: tuple[int, int, int, int] = rng_cache(key=key, n=4, low=110, high=255)
                viewer.draw_label(
                    creator=self.__class__.__name__,
                    text=label,
                    color=Color(color),
                    x=position[0] + dict_index,
                    y=position[1] + (dict_index * 8),
                    height=20,
                    width=20,
                    font_size=self.font_size
                )
