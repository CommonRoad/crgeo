from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.defaults import DEFAULT_OBSTACLE_COLOR
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


class RenderObstaclesEgoFramePlugin(BaseRendererPlugin):
    def __init__(
        self,
        color: T_ColorTuple = DEFAULT_OBSTACLE_COLOR,
        radius: float = 0.8,
        linewidth: float = 0.5
    ) -> None:
        self._color = color
        self._radius = radius
        self._linewidth = linewidth

    def __call__(self, viewer: Viewer2D, params: RenderParams) -> None:
        for index in range(params.data.v['num_nodes']):
            is_ego_vehicle = params.data.v.is_ego_mask[index].numpy().item()
            if is_ego_vehicle:
                continue
            ego_frame_rel_pos = params.data.v.pos_ego_frame[index].numpy()
            abs_pos = params.data.v.pos[index].numpy()
            translated_ego_frame_pos = params.ego_vehicle.state.position + ego_frame_rel_pos

            viewer.draw_circle(
                origin=translated_ego_frame_pos,
                radius=self._radius,
                color=self._color,
                filled=True
            )

            viewer.draw_line(
                start=abs_pos,
                end=translated_ego_frame_pos,
                linewidth=self._linewidth,
                color=self._color
            )
