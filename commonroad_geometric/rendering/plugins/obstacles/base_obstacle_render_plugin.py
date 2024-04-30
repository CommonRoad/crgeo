from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletGraphPlugin, RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer

T_ColorRandomization = Literal["obstacle", "lanelet", "viewer"]


@dataclass
class BaseRenderObstaclePlugin(BaseRenderPlugin, ABC):
    obstacle_color: Color = Color((0.1, 0.8, 0.1), alpha=1.0)
    obstacle_fill_color: Optional[Color] = None
    filled: bool = True  # Remove and substitute with obstacle_fill_color is not None, when ColorTheme is implemented
    randomize_color_from: Optional[T_ColorRandomization] = None
    obstacle_rng_cache = CachedRNG(np.random.default_rng(seed=0).random)
    obstacle_line_width: Optional[float] = None
    ignore_obstacle_ids = set()
    reference_vehicle_idx: Optional[int] = None
    skip_ego_id: bool = True
    obstacle_height = 0.0

    def get_active_rng_cache(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> CachedRNG:
        match self.randomize_color_from:
            case "lanelet":
                # Hack for synchronizing colors, taking CachedRNG from lanelet network renderer
                render_lanelets_plugins = [
                    plugin
                    for plugin in params.render_kwargs['plugins']
                    if isinstance(plugin, (RenderLaneletNetworkPlugin, RenderLaneletGraphPlugin))
                ]
                if render_lanelets_plugins:
                    return render_lanelets_plugins[0].lanelet_rng_cache
                return self.obstacle_rng_cache
            case "viewer":
                return viewer.shared_rng_cache
            case "obstacle" | _:
                return self.obstacle_rng_cache
