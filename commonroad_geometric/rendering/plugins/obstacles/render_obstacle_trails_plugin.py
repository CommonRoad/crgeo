from dataclasses import dataclass

from commonroad_geometric.common.io_extensions.obstacle import get_state_list
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


@dataclass
class RenderObstacleTrailPlugin(BaseRenderObstaclePlugin):
    trail_interval: int = 35
    num_trail_shadows: int = 5
    trail_linewidth: float = 0.5
    trail_alpha: bool = True
    trail_arrows: bool = False
    fill_shadows: bool = False

    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ):
        for obstacle in params.simulation.current_obstacles:
            if self.skip_ego_id and obstacle.obstacle_id < 0:
                continue
            if obstacle.obstacle_id in self.ignore_obstacle_ids:
                continue

            vertices = obstacle.obstacle_shape.vertices

            color = self.obstacle_color
            if self.randomize_color_from is not None:
                rng_cache = self.get_active_rng_cache(viewer, params)
                match self.randomize_color_from:
                    case "obstacle":
                        rbg_tuple: tuple[float, float, float] = rng_cache(
                            key=obstacle.obstacle_id,
                            n=3
                        )
                        color = Color(rbg_tuple)
                    case "lanelet" | "viewer" | _:
                        obstacle_lanelet_ids = params.simulation.obstacle_id_to_lanelet_id[obstacle.obstacle_id]
                        if obstacle_lanelet_ids:
                            obstacle_lanelet_id = obstacle_lanelet_ids[0]
                            rbg_tuple: tuple[float, float, float] = rng_cache(
                                key=obstacle_lanelet_id,
                                n=3
                            )
                            color = Color(rbg_tuple)

            state_list = get_state_list(obstacle, upper=params.time_step)
            index = len(state_list) - 1
            states_to_draw = 0

            for _ in range(self.num_trail_shadows):
                if index - self.trail_interval < 0:
                    break
                index -= self.trail_interval
                states_to_draw += 1

            if states_to_draw > 0:
                i = 0
                while index < len(state_list) and i < states_to_draw:
                    state_prev = state_list[index]
                    iter_perc = i / states_to_draw

                    if self.trail_alpha:
                        fill_color_trail = color.with_alpha(alpha=0.2 + 0.3 * iter_perc)
                        border_color_trail = color
                    else:
                        fill_color_trail = Color((iter_perc, color.green, iter_perc, 1.0))
                        border_color_trail = fill_color_trail

                    viewer.draw_2d_shape(
                        creator=self.__class__.__name__,
                        vertices=vertices,
                        fill_color=fill_color_trail if self.fill_shadows else None,
                        border_color=border_color_trail,
                        translation=state_prev.position,
                        rotation=state_prev.orientation,
                        line_width=self.trail_linewidth
                    )

                    i += 1
                    index += self.trail_interval
