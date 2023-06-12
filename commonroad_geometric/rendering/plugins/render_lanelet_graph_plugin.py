from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletEdgeTypeColorMap
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D

_LaneletEdgeTypeEnabledMap = {
    LaneletEdgeType.PREDECESSOR: False,
    LaneletEdgeType.SUCCESSOR: True,
    LaneletEdgeType.ADJACENT_LEFT: True,
    LaneletEdgeType.ADJACENT_RIGHT: True,
    LaneletEdgeType.OPPOSITE_LEFT: False,
    LaneletEdgeType.OPPOSITE_RIGHT: False,
    LaneletEdgeType.DIAGONAL_LEFT: False,
    LaneletEdgeType.DIAGONAL_RIGHT: False,
    LaneletEdgeType.DIVERGING: True,
    LaneletEdgeType.MERGING: True,
    LaneletEdgeType.CONFLICTING: True,
    LaneletEdgeType.CONFLICT_LINK: True,
}


@dataclass
class RenderLaneletGraphStyle:
    arrows: bool = False
    color_map: Optional[Dict[LaneletEdgeType, T_ColorTuple]] = None
    dashed: bool = False
    draw_index: bool = False
    edge_color: T_ColorTuple = (0.75, 0.75, 0.75, 1.0)
    edge_linewidth: float = 0.3
    edge_arc: float = 0.0
    enabled_map: Optional[Dict[LaneletEdgeType, bool]] = None
    multi_color: bool = False
    node_fillcolor: Optional[T_ColorTuple] = (0.75, 0.75, 0.75, 1.0)
    node_linecolor: Optional[T_ColorTuple] = (0.3, 0.3, 0.3, 0.8)
    node_linewidth: float = 0.05
    node_radius: float = 0.55
    offset_x: float = 0.0
    offset_y: float = 0.0
    right_align_laterally: bool = True


class RenderLaneletGraphPlugin(BaseRendererPlugin):
    def __init__(
        self,
        **kwargs
    ) -> None:
        self.style = RenderLaneletGraphStyle(**kwargs)

        self._offset = np.array([self.style.offset_x, self.style.offset_y])
        self._enabled_map = _LaneletEdgeTypeEnabledMap.copy()
        self._enabled_map.update({} if self.style.enabled_map is None else self.style.enabled_map)
        self._color_map = LaneletEdgeTypeColorMap.copy()
        self._color_map.update({} if self.style.color_map is None else self.style.color_map)
        self._rng_cache = CachedRNG(np.random.uniform)

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return self._render_from_scenario(viewer, params)
  
        edge_index = params.data.l2l.edge_index
        
        if self.style.right_align_laterally:
            pos = params.data.l.right_pos
        else:
            pos = params.data.l.center_pos
        pos = pos.numpy()

        # pos = [] # TODO
        # for lidx, lid in enumerate(params.data.l.id):
        #     cl = params.ego_vehicle_simulation.simulation.get_lanelet_center_polyline(lid.item())
        #     pos_i = cl.lateral_translate_point(cl.length/2, lateral_distance=3.0)
        #     pos.append(pos_i)
        
        #left_vertex = params.data.l.left_vertices[0].numpy().reshape(-1, 2)
        #right_vertex = params.data.l.right_vertices[0].numpy().reshape(-1, 2)
        for idx in range(edge_index.shape[1]):
            # Replace idx here with one from indices
            from_lanelet_idx, to_lanelet_idx = params.data.l2l.edge_index[:, idx]
            from_lanelet_pos = pos[from_lanelet_idx] 
            to_lanelet_pos = pos[to_lanelet_idx]

            
            # Select a random x position so that the connections don't overlap
            #from_lanelet_pos[0] = self._rng_cache(key=to_lanelet_idx, low=left_vertex[0][0], high=right_vertex[-1][0])
            lanelet_edge_type = params.data.l2l.type[idx].item()
            if self._enabled_map[lanelet_edge_type]:
                color = self._color_map[lanelet_edge_type] if self.style.multi_color else self.style.edge_color
                if self.style.arrows:
                    viewer.draw_arrow_from_to(
                        from_lanelet_pos + self._offset,
                        to_lanelet_pos + self._offset,
                        linewidth=self.style.edge_linewidth,
                        color=color,
                        dashed=self.style.dashed,
                        scale=0.3,
                        arc=self.style.edge_arc
                    )
                elif self.style.dashed:
                    viewer.draw_dashed_line(
                        from_lanelet_pos + self._offset,
                        to_lanelet_pos + self._offset,
                        linewidth=self.style.edge_linewidth,
                        color=color,
                        arc=self.style.edge_arc
                    )
                else:
                    viewer.draw_line(
                        from_lanelet_pos + self._offset,
                        to_lanelet_pos + self._offset,
                        linewidth=self.style.edge_linewidth,
                        color=color,
                        arc=self.style.edge_arc
                    )

        for idx, center_pos in enumerate(pos):
            # lanelet_id = params.data.lanelet.id[idx]
            # if params.ego_vehicle is not None and params.ego_vehicle.ego_route is not None and lanelet_id in params.ego_vehicle.ego_route.lanelet_id_route[:5]:
            #     color = (0.0, 0.9, 0.0)
            #     radius = self.style.node_radius
            # else:
            #     color = self.style.node_color
            #     radius = self.style.node_radius
            viewer.draw_circle(
                origin=center_pos + self._offset,
                radius=self.style.node_radius,
                filled=self.style.node_fillcolor is not None,
                linecolor=self.style.node_linecolor,
                color=self.style.node_fillcolor,
                linewidth=self.style.node_linewidth
            )

    def _render_from_scenario(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:

        for lanelet in params.scenario.lanelet_network.lanelets:
            from_lanelet_pos = lanelet.center_vertices[lanelet.center_vertices.shape[0] // 2]
    
            def draw_line(to: int, color: T_ColorTuple) -> None:
                color = color if self.style.multi_color else self.style.edge_color
                other = params.scenario.lanelet_network.find_lanelet_by_id(to)
                to_lanelet_pos = other.center_vertices[other.center_vertices.shape[0] // 2]
                viewer.draw_line(
                    from_lanelet_pos + self._offset,
                    to_lanelet_pos + self._offset,
                    linewidth=self.style.edge_linewidth,
                    color=color,
                    arc=self.style.edge_arc
                )

            for lid in lanelet.successor:
                draw_line(lid, LaneletEdgeTypeColorMap[LaneletEdgeType.SUCCESSOR])
            if lanelet.adj_left is not None:
                if lanelet.adj_left_same_direction:
                    draw_line(lanelet.adj_left, LaneletEdgeTypeColorMap[LaneletEdgeType.ADJACENT_LEFT])
                else:
                    draw_line(lanelet.adj_left, LaneletEdgeTypeColorMap[LaneletEdgeType.OPPOSITE_LEFT])
            if lanelet.adj_right is not None:
                if lanelet.adj_right_same_direction:
                    draw_line(lanelet.adj_right, LaneletEdgeTypeColorMap[LaneletEdgeType.ADJACENT_RIGHT])
                else:
                    draw_line(lanelet.adj_right, LaneletEdgeTypeColorMap[LaneletEdgeType.OPPOSITE_RIGHT])


        for lanelet in params.scenario.lanelet_network.lanelets:
            from_lanelet_pos = lanelet.center_vertices[lanelet.center_vertices.shape[0] // 2]

            viewer.draw_circle(
                origin=from_lanelet_pos + self._offset,
                radius=self.style.node_radius,
                filled=True,
                color=(0.0, 1.0, 0.0),
                linewidth=0.3
            )
