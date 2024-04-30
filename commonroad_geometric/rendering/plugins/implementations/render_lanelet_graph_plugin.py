from dataclasses import dataclass, field

import numpy as np

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletEdgeTypeColorMap
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer

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
class RenderLaneletGraphPlugin(BaseRenderPlugin):
    node_fillcolor: Color = Color((0.75, 0.75, 0.75), alpha=1.0)
    node_linecolor: Color = Color((0.3, 0.3, 0.3), alpha=0.8)
    node_radius: float = 0.55
    node_linewidth: float = 0.05
    offset_x: float = 0.0
    offset_y: float = 0.0
    right_align_laterally: bool = True
    edge_color: Color = Color((0.75, 0.75, 0.75), alpha=1.0)
    edge_linewidth: float = 0.3
    edge_arc: float = 0.0  # Not implemented in BaseViewers currently
    arrows: bool = False
    dashed: bool = False
    draw_index: bool = False
    multi_color: bool = False
    color_map: dict[LaneletEdgeType, Color] = field(default_factory=LaneletEdgeTypeColorMap.copy)
    enabled_map: dict[LaneletEdgeType, bool] = field(default_factory=_LaneletEdgeTypeEnabledMap.copy)
    lanelet_rng_cache = CachedRNG(np.random.default_rng(seed=0).random)

    @property
    def offset(self) -> np.ndarray:
        return np.array([self.offset_x, self.offset_y])

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return self._render_from_scenario(viewer, params)

        edge_index = params.data.l2l.edge_index

        if self.right_align_laterally:
            pos = params.data.l.right_pos
        else:
            pos = params.data.l.center_pos
        pos = pos.numpy()

        # pos = [] # TODO
        # for lidx, lid in enumerate(params.data.l.id):
        #     cl = params.ego_vehicle_simulation.simulation.get_lanelet_center_polyline(lid.item())
        #     pos_i = cl.lateral_translate_point(cl.length/2, lateral_distance=3.0)
        #     pos.append(pos_i)

        # left_vertex = params.data.l.left_vertices[0].numpy().reshape(-1, 2)
        # right_vertex = params.data.l.right_vertices[0].numpy().reshape(-1, 2)
        for idx in range(edge_index.shape[1]):
            # Replace idx here with one from indices
            from_lanelet_idx, to_lanelet_idx = params.data.l2l.edge_index[:, idx]
            from_lanelet_pos = pos[from_lanelet_idx]
            to_lanelet_pos = pos[to_lanelet_idx]

            # Select a random x position so that the connections don't overlap
            # from_lanelet_pos[0] = self._rng_cache(key=to_lanelet_idx, low=left_vertex[0][0], high=right_vertex[-1][0])
            lanelet_edge_type = params.data.l2l.type[idx].item()
            if self.enabled_map[lanelet_edge_type]:
                color = self.color_map[lanelet_edge_type] if self.multi_color else self.edge_color
                if self.arrows:
                    viewer.draw_arrow_from_to(
                        creator=self.__class__.__name__,
                        start=from_lanelet_pos + self.offset,
                        end=to_lanelet_pos + self.offset,
                        line_color=color,
                        line_width=self.edge_linewidth,
                        arrow_head_size=0.3,
                        arc=self.edge_arc
                    )
                elif self.dashed:
                    viewer.draw_dashed_line(
                        creator=self.__class__.__name__,
                        start=from_lanelet_pos + self.offset,
                        end=to_lanelet_pos + self.offset,
                        color=color,
                        line_width=self.edge_linewidth,
                        arc=self.edge_arc
                    )
                else:
                    viewer.draw_line(
                        creator=self.__class__.__name__,
                        start=from_lanelet_pos + self.offset,
                        end=to_lanelet_pos + self.offset,
                        color=color,
                        line_width=self.edge_linewidth,
                        arc=self.edge_arc
                    )

        for idx, center_pos in enumerate(pos):
            # lanelet_id = params.data.lanelet.id[idx]
            # if params.ego_vehicle is not None and params.ego_vehicle.ego_route is not None and lanelet_id in params.ego_vehicle.ego_route.lanelet_id_route[:5]:
            #     color = (0.0, 0.9, 0.0)
            #     radius = self.node_radius
            # else:
            #     color = self.node_color
            #     radius = self.node_radius
            viewer.draw_circle(
                creator=self.__class__.__name__,
                origin=center_pos + self.offset,
                radius=self.node_radius,
                fill_color=self.node_fillcolor,
                border_color=self.node_linecolor,
                line_width=self.node_linewidth
            )

    def _render_from_scenario(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:

        for lanelet in params.scenario.lanelet_network.lanelets:
            from_lanelet_pos = lanelet.center_vertices[lanelet.center_vertices.shape[0] // 2]

            def draw_line(to: int, color: Color) -> None:
                color_ = color if self.multi_color else self.edge_color
                other = params.scenario.lanelet_network.find_lanelet_by_id(to)
                to_lanelet_pos = other.center_vertices[other.center_vertices.shape[0] // 2]
                viewer.draw_line(
                    creator=self.__class__.__name__,
                    start=from_lanelet_pos + self.offset,
                    end=to_lanelet_pos + self.offset,
                    color=color_,
                    line_width=self.edge_linewidth,
                    arc=self.edge_arc
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

            green = Color("green")
            viewer.draw_circle(
                creator=self.__class__.__name__,
                origin=from_lanelet_pos + self.offset,
                radius=self.node_radius,
                fill_color=green,
                border_color=green,
                line_width=0.3
            )
