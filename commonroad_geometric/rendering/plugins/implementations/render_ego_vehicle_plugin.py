from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Optional

import numpy as np

from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderEgoVehiclePlugin(BaseRenderPlugin):
    ego_vehicle_vertices: np.ndarray = np.array([
        [-2.4, -1.],
        [-2.4, 1.],
        [2.4, 1.],
        [2.4, -1.],
        [-2.4, -1.]
    ])
    ego_vehicle_linewidth: Optional[float] = None
    ego_vehicle_fill_color: Optional[Color] = None
    ego_vehicle_color: Optional[Color] = None
    ego_vehicle_color_collision: Optional[Color] = Color((1.0, 0.0, 0.0, 1.0))
    render_trail: bool = False
    direction_arrow: bool = False
    trail_interval: int = 35
    num_trail_shadows: int = 5
    trail_linewidth: float = 0.5
    trail_alpha: bool = True
    trail_arrows: bool = False
    fill_shadows: bool = True
    filled: bool = True
    trail_velocity_profile: bool = False

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            return

        if self.ego_vehicle_color is None:
            if viewer.options.theme == ColorTheme.DARK:
                self.ego_vehicle_color = Color((1.0, 1.0, 1.0, 1.0))
            else:
                self.ego_vehicle_color = Color((0.1, 0.8, 0.1, 1.0))

        if self.ego_vehicle_fill_color is None:
            if viewer.options.theme == ColorTheme.DARK:
                self.ego_vehicle_fill_color = Color((0.0, 0.0, 0.0, 1.0))
            else:
                self.ego_vehicle_fill_color = Color((0.6, 1.0, 0.6, 1.0))

        if self.ego_vehicle_linewidth is None:
            if viewer.options.theme == ColorTheme.DARK:
                self.ego_vehicle_linewidth = 1.0
            else:
                self.ego_vehicle_linewidth = 0.5

        if self.trail_velocity_profile:
            for index in range(len(params.ego_vehicle.state_list) - 1):
                state = params.ego_vehicle.state_list[index]
                position = state.position
                velocity = min(20.0, state.velocity)
                radius = min(1.0, 1.2 / sqrt(velocity + 0.1))
                alpha = min(0.1, min(1.0, 0.7 / sqrt(velocity + 0.01)))

                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=position,
                    radius=radius,
                    fill_color=Color((0.1, 0.9, 0.1), alpha=alpha),
                )

        if self.render_trail:
            index = len(params.ego_vehicle.state_list) - 1
            states_to_draw = 0

            for _ in range(self.num_trail_shadows):
                if index - self.trail_interval < 0:
                    break
                index -= self.trail_interval
                states_to_draw += 1

            if states_to_draw > 0:
                iteration = 0
                while index < len(params.ego_vehicle.state_list):
                    state = params.ego_vehicle.state_list[index]
                    percentage = iteration / states_to_draw

                    if self.trail_alpha:
                        fill_color_trail = self.ego_vehicle_fill_color.with_alpha(alpha=0.2 + 0.3 * percentage)
                        border_color_trail = self.ego_vehicle_color
                    else:
                        fill_color_trail = Color((percentage, self.ego_vehicle_fill_color.green, percentage), alpha=1.0)
                        border_color_trail = Color((percentage, self.ego_vehicle_color.green, percentage), alpha=1.0)

                    viewer.draw_2d_shape(
                        creator=self.__class__.__name__,
                        vertices=self.ego_vehicle_vertices,
                        fill_color=fill_color_trail if self.fill_shadows else None,
                        border_color=border_color_trail,
                        translation=state.position,
                        rotation=state.orientation,
                        line_width=self.trail_linewidth
                    )

                    if self.trail_arrows:
                        yaw = state.orientation
                        p_front = state.position + params.ego_vehicle.parameters.l / 2 * np.array([cos(yaw), sin(yaw)])
                        ego_arrow_length = state.velocity / 3
                        viewer.draw_2d_arrow(
                            creator=self.__class__.__name__,
                            origin=p_front,
                            angle=state.orientation,
                            length=ego_arrow_length,
                            line_color=Color("green", alpha=border_color_trail.alpha),
                            line_width=1.0,
                            arrow_head_size=1.0
                        )

                    iteration += 1

                    index += self.trail_interval

        if self.ego_vehicle_color_collision is not None and params.ego_vehicle_simulation.check_if_has_collision().collision: 
            border_color = self.ego_vehicle_color_collision
        else:
            border_color = self.ego_vehicle_color

        viewer.draw_2d_shape(
            creator=self.__class__.__name__,
            vertices=self.ego_vehicle_vertices,
            fill_color=self.ego_vehicle_fill_color if self.filled else None,
            border_color=border_color,
            translation=params.ego_vehicle.state.position,
            rotation=params.ego_vehicle.state.orientation,
            line_width=self.ego_vehicle_linewidth
        )

        if self.direction_arrow:
            yaw = params.ego_vehicle.state.orientation
            p_front = params.ego_vehicle.state.position # + params.ego_vehicle.parameters.l / 2 * np.array([cos(yaw), sin(yaw)])
            ego_arrow_length = 20 + params.ego_vehicle.state.velocity / 3
            viewer.draw_2d_arrow(
                creator=self.__class__.__name__,
                origin=p_front,
                angle=yaw,
                length=ego_arrow_length,
                line_color=Color("red"),
                line_width=1.0,
                arrow_head_size=1.0
            )