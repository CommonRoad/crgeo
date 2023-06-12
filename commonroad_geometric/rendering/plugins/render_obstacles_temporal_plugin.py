import colorsys
from typing import cast

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.defaults import DEFAULT_OBSTACLE_COLOR, DEFAULT_OBSTACLE_LINEWIDTH
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


class RenderObstaclesTemporalPlugin(BaseRendererPlugin):
    def __init__(
        self,
        obstacle_linewidth: float = DEFAULT_OBSTACLE_LINEWIDTH,
        obstacle_fill_color: T_ColorTuple = (0, 0, 0),
        obstacle_color: T_ColorTuple = DEFAULT_OBSTACLE_COLOR,
        max_prev_time_steps: int = 5,
        randomize_color: bool = False,
        draw_index: bool = False,
        font_size: float = 7.0
    ) -> None:
        self._obstacle_linewidth = obstacle_linewidth
        self._obstacle_fill_color = obstacle_fill_color
        self._obstacle_color = obstacle_color
        self._max_prev_time_steps = max_prev_time_steps
        self._randomize_color = randomize_color
        self._draw_index = draw_index
        self._font_size = font_size
        self._rng_cache = CachedRNG(np.random.random_sample) if randomize_color else None

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        self._obstacle_linewidth = params.render_kwargs.pop("obstacle_linewidth", self._obstacle_linewidth)
        self._obstacle_fill_color = params.render_kwargs.pop("obstacle_fill_color", self._obstacle_fill_color)
        self._obstacle_color = params.render_kwargs.pop("obstacle_color", self._obstacle_color)
        self._draw_index = params.render_kwargs.pop("draw_index", self._draw_index)
        self._font_size = params.render_kwargs.pop("font_size", self._font_size)

        assert params.data is not None
        data = cast(CommonRoadDataTemporal, params.data)

        max_t_offset = min(self._max_prev_time_steps, params.time_step - data.time_step.min().item())
        for t_offset in range(-max_t_offset, 1):
            # vehicle nodes at the current time step
            vehicle_data_t = data.vehicle_at_time_step(max_t_offset + t_offset)

            for vehicle_idx in range(vehicle_data_t.x.size(0)):
                if self._randomize_color:
                    hue = self._rng_cache(key=vehicle_data_t.id[vehicle_idx].item())
                    color = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                else:
                    color = self._obstacle_color

                # t_offset = 0 is opaque, transparency increases with -t_offset
                color = (*color[:3], 1.0 + t_offset / max(1, max_t_offset) * 0.9)

                # only draw labels for vehicle nodes at the current simulation time step
                if t_offset == 0 and self._draw_index:
                    label = str(vehicle_data_t.indices[vehicle_idx].numpy())
                    if "labels" in params.render_kwargs:
                        label = f"{params.render_kwargs['labels'][vehicle_idx]:.2f}"
                else:
                    label = None

                vertices = vehicle_data_t.vertices[vehicle_idx].view(-1, 2).numpy()
                viewer.draw_shape(
                    vertices,
                    vehicle_data_t.pos[vehicle_idx].numpy(),
                    vehicle_data_t.orientation[vehicle_idx].item(),
                    filled=True,
                    linewidth=self._obstacle_linewidth,
                    fill_color=self._obstacle_fill_color,
                    border=color,
                    label=label,
                    label_color=(255, 255, 255, 255),
                    font_size=self._font_size,
                    label_height=20,
                    label_width=20,
                )
