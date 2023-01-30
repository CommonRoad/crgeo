import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import torch


from crgeo.common.caching.cached_rng import CachedRNG
from crgeo.common.io_extensions.obstacle import get_state_list, state_at_time
from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin, T_OutputTransform
from crgeo.rendering.defaults import ColorTheme, DEFAULT_OBSTACLE_COLOR
from crgeo.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from crgeo.rendering.types import RenderParams
from crgeo.rendering.color_utils import T_ColorTuple
from crgeo.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderObstaclesStyle:
    obstacle_linewidth: Optional[float] = None
    obstacle_fill_color: Optional[T_ColorTuple] = None
    obstacle_color: Optional[T_ColorTuple] = None
    from_graph: bool = False
    filled: bool = True
    graph_vertices_attr: Optional[str] = "vertices"
    graph_position_attr: Optional[str] = "pos"
    graph_orientation_attr: Optional[str] = "orientation"
    randomize_color: bool = False
    randomize_color_from_lanelet: bool = False
    draw_index: bool = False
    font_size: float = 7.0
    transforms: List[T_OutputTransform] = None
    render_trail: bool = False
    trail_interval: int = 35
    num_trail_shadows: int = 5
    trail_linewidth: float = 0.5
    trail_alpha: bool = True
    trail_arrows: bool = False
    fill_shadows: bool = False


class RenderObstaclesPlugin(BaseRendererPlugin):
    def __init__(
        self,
        style: Optional[RenderObstaclesStyle] = None,
        **kwargs
    ) -> None:
        self.style = style if style is not None else RenderObstaclesStyle(**kwargs)
        assert not (self.style.randomize_color and self.style.randomize_color_from_lanelet)
        self._rng_cache: Optional[CachedRNG] = None

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:

        if self.style.randomize_color_from_lanelet:
            # hack for synchronizing colors, taking CachedRNG from lanelet network renderer
            render_lanelets_plugins = [plugin for plugin in params.render_kwargs['plugins'] if isinstance(plugin, RenderLaneletNetworkPlugin)]
            if len(render_lanelets_plugins) > 0:
                self._rng_cache = render_lanelets_plugins[0]._rng_cache

        if self._rng_cache is None and (self.style.randomize_color or self.style.randomize_color_from_lanelet):
            self._rng = np.random.default_rng(seed=0)
            self._rng_cache = CachedRNG(self._rng.random)

        if params.render_kwargs is not None:
            obstacle_linewidth = params.render_kwargs.pop('obstacle_linewidth', self.style.obstacle_linewidth)
            obstacle_fill_color = params.render_kwargs.pop('obstacle_fill_color', self.style.obstacle_fill_color)
            obstacle_color = params.render_kwargs.pop('obstacle_color', self.style.obstacle_color)

            self.style.from_graph = params.render_kwargs.pop('from_graph', self.style.from_graph)
            self.style.randomize_color = params.render_kwargs.pop('randomize_color', self.style.randomize_color)
            self.style.draw_index = params.render_kwargs.pop('draw_index', self.style.draw_index)
            self.style.font_size = params.render_kwargs.pop('font_size', self.style.font_size)

        if obstacle_fill_color is None:
            if viewer.theme == ColorTheme.BRIGHT:
                # obstacle_fill_color = (0.6, 1.0, 0.6, 1.0) 
                obstacle_fill_color = (0.8, 0.8, 0.8, 1.0)
            else:
                obstacle_fill_color = (0.6, 1.0, 0.6, 0.0)
        if obstacle_color is None:
            if viewer.theme == ColorTheme.BRIGHT:
                # obstacle_color = (0.9, 0.0, 0.0, 1.0)
                obstacle_color = (0.2, 0.2, 0.2, 1.0)
                # obstacle_color = (0.0, 0.9, 0.0, 1.0)
            else:
                obstacle_color = DEFAULT_OBSTACLE_COLOR
        if obstacle_linewidth is None:
            if viewer.theme == ColorTheme.BRIGHT:
                obstacle_linewidth = 0.5
            else:
                obstacle_linewidth = 0.7

        style_opt = asdict(self.style)
        style_opt.update(dict(
            obstacle_linewidth=obstacle_linewidth,
            obstacle_fill_color=obstacle_fill_color,
            obstacle_color=obstacle_color,
        ))
        style = RenderObstaclesStyle(**style_opt)

        self._ignore_obstacle_ids = params.render_kwargs.pop('ignore_obstacle_ids', False)

        if self.style.render_trail:
            self._render_trails(viewer, params, style)

        if self.style.from_graph:
            self._render_from_graph(viewer, params, style)
        else:
            self._render_from_scenario(viewer, params, style)

    def _render_trails(self, viewer: Viewer2D, params: RenderParams, style: RenderObstaclesStyle):
        for obstacle in params.scenario.dynamic_obstacles:
            if obstacle.obstacle_id < 0: # TODO: ego vehicle shown?
                continue
            if self._ignore_obstacle_ids and obstacle.obstacle_id in self._ignore_obstacle_ids:
                continue

            state = state_at_time(obstacle, params.time_step)
            if state is None:
                continue
            vertices = obstacle.obstacle_shape.vertices
            if style.randomize_color:
                color = self._rng_cache(
                    key=obstacle.obstacle_id,
                    n=3
                )
            elif style.randomize_color_from_lanelet:
                color = self._rng_cache(
                    key=params.simulation.obstacle_id_to_lanelet_id[obstacle.obstacle_id][0],
                    n=3
                )
            else:
                color = style.obstacle_color

            state_list = get_state_list(obstacle, upper=params.time_step)
            index = len(state_list) - 1
            states_to_draw = 0

            for iter in range(style.num_trail_shadows):
                if index - style.trail_interval < 0:
                    break
                index -= style.trail_interval
                states_to_draw += 1

            if states_to_draw > 0:
                iter = 0
                while index < len(state_list):
                    state_prev = state_list[index]
                    iter_perc = iter / (states_to_draw)

                    if style.trail_alpha:
                        fill_color_trail = list(color)
                        border_color_trail = list(color)
                        fill_color_trail[-1] = 0.2 + 0.6*0.5*iter_perc
                        # border_color_trail[-1] = 0.2 + 0.6*(1 - iter_perc)
                    else:
                        border_color_trail = list(color)
                        fill_color_trail = [iter_perc, color[1], iter_perc, 1.0]
                        border_color_trail = [iter_perc, border_color_trail[1], iter_perc, 1.0]

                    viewer.draw_shape(
                        vertices,
                        state_prev.position,
                        state_prev.orientation,
                        color=border_color_trail,
                        filled=style.fill_shadows,
                        linewidth=style.trail_linewidth,
                        fill_color=fill_color_trail if style.fill_shadows else None,
                        border=border_color_trail
                    )

                    iter += 1
                    index += style.trail_interval

    def _render_from_scenario(self, viewer: Viewer2D, params: RenderParams, style: RenderObstaclesStyle) -> None:
        for obstacle in params.scenario.dynamic_obstacles:
            if obstacle.obstacle_id < 0: # TODO: ego vehicle shown?
                continue
            if self._ignore_obstacle_ids and obstacle.obstacle_id in self._ignore_obstacle_ids:
                continue

            state = state_at_time(obstacle, params.time_step)
            if state is None:
                # Obstacle not present at this time-step
                continue
            vertices = obstacle.obstacle_shape.vertices
            if style.randomize_color:
                color = self._rng_cache(
                    key=obstacle.obstacle_id,
                    n=3
                )
                fill_color = (color[0]*0.7, color[1]*0.7, color[2]*0.7)
            elif style.randomize_color_from_lanelet:
                color = self._rng_cache(
                    key=params.simulation.obstacle_id_to_lanelet_id[obstacle.obstacle_id][0],
                    n=3
                )
                fill_color = style.obstacle_fill_color
            else:
                color = style.obstacle_color
                fill_color = style.obstacle_fill_color

            viewer.draw_shape(
                vertices,
                state.position,
                state.orientation,
                color=color,
                filled=style.filled,
                linewidth=style.obstacle_linewidth,
                fill_color=fill_color,
                border=color,
                label=str(obstacle.obstacle_id) if style.draw_index else None,
                label_color=(255, 255, 255, 255),
                font_size=style.font_size,
                label_height=20,
                label_width=20,
            )

    def _render_from_graph(self, viewer: Viewer2D, params: RenderParams, style: RenderObstaclesStyle) -> None:
        if params.data is None:
            warnings.warn(f"Graph data not included in rendering params - cannot render obstacles from graph.")
            return

        if style.transforms is not None:
            for transform in style.transforms:
                params = transform(params)

        for index in range(params.data.v['num_nodes']):
            obstacle_id = params.data.v.id[index].item()
            if self._ignore_obstacle_ids and obstacle_id in self._ignore_obstacle_ids:
                continue

            if style.randomize_color:
                color = self._rng_cache(
                    key=obstacle_id,
                    n=3
                )
            elif style.randomize_color_from_lanelet:
                obstacle_lanelet_assignments = torch.where(params.data.v2l.edge_index[0, :] == index)[0]
                if len(obstacle_lanelet_assignments) > 0:
                    obstacle_lanelet_edge_idx = obstacle_lanelet_assignments[0].item()
                    obstacle_lanelet_idx = params.data.v2l.edge_index[1, obstacle_lanelet_edge_idx].item()
                    obstacle_lanelet_id = params.data.l.id[obstacle_lanelet_idx].item()
                    color = self._rng_cache(
                        key=obstacle_lanelet_id,
                        n=3
                    )
                else:
                    color = style.obstacle_color
            else:
                color = style.obstacle_color
            
            # if params.render_kwargs is not None and 'labels' in params.render_kwargs:
            #     label = str(f"{params.render_kwargs['labels'][index]:.2f}")
            # else:
            #     label = str(params.data.v.indices[index].numpy())

            if style.graph_vertices_attr is not None:
                vertices = torch.reshape(params.data.v[style.graph_vertices_attr][index], (-1, 2)).numpy()
                viewer.draw_shape(
                    vertices,
                    params.data.v[style.graph_position_attr][index].numpy(),
                    params.data.v[style.graph_orientation_attr][index],
                    color=color,
                    filled=style.filled,
                    linewidth=style.obstacle_linewidth,
                    fill_color=style.obstacle_fill_color,
                    border=color,
                    #label=label if style.draw_index else None,
                    label_color=(255, 255, 255, 255),
                    font_size=style.font_size,
                    label_height=20,
                    label_width=20,
                )
            else:
                viewer.draw_circle(
                    origin=params.data.v[style.graph_position_attr][index].numpy(),
                    radius=5,
                    color=color,
                    linewidth=style.obstacle_linewidth,
                    linecolor=color,
                    # label=label if style.draw_index else None,
                    label_color=(255, 255, 255, 255),
                    font_size=style.font_size,
                )
