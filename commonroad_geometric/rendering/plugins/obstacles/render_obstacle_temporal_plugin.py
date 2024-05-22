import colorsys
from dataclasses import dataclass

from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderObstacleTemporalPlugin(BaseRenderObstaclePlugin):
    max_prev_time_steps: int = 5
    randomize_color: bool = False
    draw_index: bool = False
    font_size: float = 7.0

    def __post_init__(self):
        self.obstacle_fill_color = Color("black")

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        data = params.data
        if data is None:
            return
        if not isinstance(data, CommonRoadDataTemporal):
            return

        max_t_offset = min(self.max_prev_time_steps, params.time_step - data.time_step.min().item())
        for t_offset in range(-max_t_offset, 1):
            # vehicle nodes at the current time step
            vehicle_data_t = data.vehicle_at_time_step(max_t_offset + t_offset)

            for vehicle_idx in range(vehicle_data_t.x.size(0)):
                if self.randomize_color:
                    rng_cache = self.get_active_rng_cache(viewer=viewer, params=params)
                    hue = rng_cache(key=vehicle_data_t.id[vehicle_idx].item(), n=1)
                    color = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                    # t_offset = 0 is opaque, transparency increases with -t_offset
                    color = Color(*color[:3], alpha=1.0 + t_offset / max(1, max_t_offset) * 0.9)
                else:
                    color = self.obstacle_color

                # only draw labels for vehicle nodes at the current simulation time step
                if t_offset == 0 and self.draw_index:
                    label = str(vehicle_data_t.indices[vehicle_idx].numpy())
                    if "labels" in params.render_kwargs:
                        label = f"{params.render_kwargs['labels'][vehicle_idx]:.2f}"
                else:
                    label = None

                vertices = vehicle_data_t.vertices[vehicle_idx].view(-1, 2).numpy()
                viewer.draw_2d_shape(
                    creator=self.__class__.__name__,
                    vertices=vertices,
                    fill_color=self.obstacle_fill_color,
                    border_color=color,
                    translation=vehicle_data_t.pos[vehicle_idx].numpy(),
                    rotation=vehicle_data_t.orientation[vehicle_idx].item(),
                    line_width=self.obstacle_line_width,
                )
