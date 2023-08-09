
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.utils.string import numpy_prettify
from commonroad_geometric.rendering.defaults import DEFAULT_OBSTACLE_COLOR
from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.rendering.viewer.viewer_2d import TEXT_COLOR, Viewer2D
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin, T_OutputTransform
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.color_utils import T_ColorTuple


class RenderObstacleLabels(BaseRendererPlugin):
    def __init__(
        self,
        from_graph: bool = False,
        graph_position_attr: Optional[str] = "pos",
        font_size: float = 7,
        transforms: List[T_OutputTransform] = None,
        as_overlays: bool = False,
    ) -> None:
        self._transforms = transforms
        self._from_graph = from_graph
        self._graph_position_attr = graph_position_attr
        self._font_size = font_size
        self._rng_cache = CachedRNG(np.random.randint)
        self._as_overlays = as_overlays

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.render_kwargs is not None:
            self._from_graph = params.render_kwargs.pop('from_graph', self._from_graph)
            self._font_size = params.render_kwargs.pop('font_size', self._font_size)

        if self._from_graph:
            self._render_from_graph(viewer, params)
        else:
            self._render_from_scenario(viewer, params)

    def _render_from_scenario(self, viewer: Viewer2D, params: RenderParams) -> None:
        raise NotImplementedError();

    def _render_from_graph(self, viewer: Viewer2D, params: RenderParams) -> None:
        if params.data is None:
            return

        if self._transforms is not None:
            for transform in self._transforms:
                params = transform(params)

        if self._as_overlays:
            overlays = params.render_kwargs.get('overlays', None)
            overlay_headings: list = params.render_kwargs.get('overlay_headings', None)
            text = ''
            if overlay_headings is not None:
                text = ' ' + ' '.join([f"{v}" for v in overlay_headings]) + '  \n'
            text += '\n'.join([f" {k} {v} " for k, v in overlays.items()])
            viewer.add_label_to_overlays(
                label=text,
                color=TEXT_COLOR,
                font_size=10,
                x=viewer.width - 900,
                y=viewer.height - 40,
                anchor_x='left', 
                anchor_y='top', 
                width=350, 
                height=600, 
                dpi=85,
                multiline=True,
            )
        else:
            for index in range(params.data.v['num_nodes']):
                position = params.data.v[self._graph_position_attr][index].numpy()
                if params.render_kwargs is not None and 'label_dict' in params.render_kwargs:
                    if isinstance(params.render_kwargs['label_dict'], dict):
                        for dict_index, key in enumerate(params.render_kwargs['label_dict']):
                            label = str(f"{key}:{params.render_kwargs['label_dict'][key][index].cpu().detach().numpy():.1f}")
                            color=self._rng_cache(key=key,n=4,low=110,high=255)
                            viewer.draw_label(
                                x=position[0] + (dict_index), 
                                y=position[1] + (dict_index*8), 
                                label=label, 
                                color=color, 
                                height=20, 
                                width=20, 
                                font_size=self._font_size
                            )
                    else:
                        return
