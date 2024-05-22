from dataclasses import dataclass

from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderObstaclesEgoFramePlugin(BaseRenderPlugin):
    color: Color = Color((0.1, 0.8, 0.1), alpha=1.0)
    radius: float = 0.8
    linewidth: float = 0.5

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return

        for index in range(params.data.v['num_nodes']):
            is_ego_vehicle = params.data.v.is_ego_mask[index].numpy().item()
            if is_ego_vehicle:
                continue
            ego_frame_rel_pos = params.data.v.pos_ego_frame[index].numpy()
            abs_pos = params.data.v.pos[index].numpy()
            translated_ego_frame_pos = params.ego_vehicle.state.position + ego_frame_rel_pos

            viewer.draw_circle(
                creator=self.__class__.__name__,
                origin=translated_ego_frame_pos,
                radius=self.radius,
                fill_color=self.color,
                border_color=self.color
            )

            viewer.draw_line(
                creator=self.__class__.__name__,
                start=abs_pos,
                end=translated_ego_frame_pos,
                color=self.color,
                line_width=self.linewidth
            )
