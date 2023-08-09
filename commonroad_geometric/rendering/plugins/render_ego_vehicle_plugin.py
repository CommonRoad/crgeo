from dataclasses import dataclass
from typing import Optional
import numpy as np
from math import cos, sin, sqrt

from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.defaults import DEFAULT_OBSTACLE_LINEWIDTH, ColorTheme
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderEgoVehicleStyle():
    ego_vehicle_linewidth: Optional[float] = None
    ego_vehicle_fill_color: Optional[T_ColorTuple] = None
    ego_vehicle_color: Optional[T_ColorTuple] = None
    render_trail: bool = False
    trail_interval: int = 35
    num_trail_shadows: int = 5
    trail_linewidth: float = 0.5
    trail_alpha: bool = True
    trail_arrows: bool = False
    fill_shadows: bool = True
    filled: bool = True
    trail_velocity_profile: bool = False
    

class RenderEgoVehiclePlugin(BaseRendererPlugin):
    def __init__(
        self,
        style: Optional[RenderEgoVehicleStyle] = None,
        **kwargs
    ) -> None:
        self._ego_vehicle_vertices = np.array([
            [-2.4, -1.],
            [-2.4, 1.],
            [2.4, 1.],
            [2.4, -1.],
            [-2.4, -1.]
        ])
        self.style = style if style is not None else RenderEgoVehicleStyle(**kwargs)

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            return

        render_kwargs = {} if params.render_kwargs is None else params.render_kwargs

        linewidth = render_kwargs.pop('ego_vehicle_linewidth', self.style.ego_vehicle_linewidth)
        fill_color = render_kwargs.pop('ego_vehicle_fill_color', self.style.ego_vehicle_fill_color)
        ego_vertices = render_kwargs.pop('ego_vehicle_vertices', self._ego_vehicle_vertices)
        ego_color = render_kwargs.pop('ego_vehicle_color', self.style.ego_vehicle_color)

        if ego_color is None:
            if viewer.theme == ColorTheme.DARK:
                ego_color = (1.0, 1.0, 1.0, 1.0)
            else:
                ego_color = (0.1, 0.8, 0.1, 1.0)
        
        if fill_color is None:
            if viewer.theme == ColorTheme.DARK:
                fill_color = (0.0, 0.0, 0.0, 1.0)
            else:
                fill_color = (0.6, 1.0, 0.6, 1.0)

        if linewidth is None:
            if viewer.theme == ColorTheme.DARK:
                linewidth = 1.0
            else:
                linewidth = 0.5

        if self.style.trail_velocity_profile:
            for index in range(len(params.ego_vehicle.state_list) - 1):
                state = params.ego_vehicle.state_list[index]
                next_state = params.ego_vehicle.state_list[index + 1]
                pos = state.position
                velocity = min(20.0, state.velocity)
                radius = min(1.0, 1.2/sqrt(velocity + 0.1))
                alpha = min(0.1, min(1.0, 0.7/sqrt(velocity + 0.01)))

                # index_perc = index / len(params.ego_vehicle.state_list)
                
                color = (
                    0.1,
                    0.9,
                    0.1,
                    alpha
                )

                viewer.draw_circle(
                    origin=pos,
                    color=color,
                    outline=False,
                    radius=radius
                )

                # viewer.draw_line(
                #     pos,
                #     next_state.position,
                #     color=color,
                #     linewidth=5.0
                # )

        if self.style.render_trail:
            index = len(params.ego_vehicle.state_list) - 1
            states_to_draw = 0

            for iter in range(self.style.num_trail_shadows):
                if index - self.style.trail_interval < 0:
                    break
                index -= self.style.trail_interval
                states_to_draw += 1

            if states_to_draw > 0:
                iter = 0
                while index < len(params.ego_vehicle.state_list): 
                    state = params.ego_vehicle.state_list[index]
                    iter_perc = iter / (states_to_draw)

                    if self.style.trail_alpha:
                        fill_color_trail = list(fill_color)
                        border_color_trail = list(ego_color)
                        fill_color_trail[-1] = 0.2 + 0.6*0.5*iter_perc
                        # border_color_trail[-1] = 0.2 + 0.6*(1 - iter_perc)
                    else:
                        border_color_trail = list(ego_color)
                        fill_color_trail = [iter_perc, fill_color[1], iter_perc, 1.0]
                        border_color_trail = [iter_perc, border_color_trail[1], iter_perc, 1.0]
                    
                    viewer.draw_shape(
                        ego_vertices,
                        state.position,
                        state.orientation,
                        color=border_color_trail,
                        filled=self.style.fill_shadows,
                        linewidth=self.style.trail_linewidth,
                        fill_color=fill_color_trail if self.style.fill_shadows else None,
                        border=border_color_trail
                    )

                    if self.style.trail_arrows:
                        yaw = state.orientation
                        p_front = state.position + params.ego_vehicle.parameters.l/2*np.array([cos(yaw), sin(yaw)])
                        ego_arrow_length = state.velocity/3
                        viewer.draw_arrow(
                            base=p_front,
                            angle=state.orientation,
                            length=ego_arrow_length,
                            color=(0.0, 0.0, 0.0, border_color_trail[-1]),
                            linewidth=1.0,
                            scale=1.0
                        )
                    
                    iter += 1
                    index += self.style.trail_interval

        viewer.draw_shape(
            ego_vertices,
            params.ego_vehicle.state.position,
            params.ego_vehicle.state.orientation,
            color=ego_color,
            filled=self.style.filled,
            linewidth=linewidth,
            fill_color=fill_color,
            border=ego_color
        )
        # from math import cos, sin
        # yaw = params.ego_vehicle.state.orientation
        # p = params.ego_vehicle.state.position
        # l = params.ego_vehicle.parameters.l
        # p_front = p + l/2*np.array([cos(yaw), sin(yaw)])
        # viewer.draw_circle(
        #     origin=p_front,
        #     radius=0.5,
        #     color=ego_color
        # )
