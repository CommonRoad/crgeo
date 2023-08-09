from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from commonroad.scenario.lanelet import Lanelet
from torch import Tensor

from commonroad_geometric.common.caching.cached_rng import CachedRNG
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import resample_polyline
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.defaults import ColorTheme
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


class RenderLaneletNetworkPlugin(BaseRendererPlugin):
    def __init__(
        self,
        # TODO: style
        lanelet_linewidth: Optional[float] = None,
        lanelet_color: T_ColorTuple = (0.65, 0.65, 0.65, 1.0), # grey
        randomize_lanelet_color: bool = False,
        enable_ego_rendering: bool = False,
        ego_outside_map_color: T_ColorTuple = (0.2, 0.2, 0.2, 0.6), # dark gray
        ego_route_color: T_ColorTuple = (0.7, 0.7, 0.7, 1.0), # light graw
        ego_lanelet_color: T_ColorTuple = (0.5, 0.5, 0.9, 1.0), # blue
        ego_conflict_color: T_ColorTuple = (0.7, 0.1, 0.1, 1.0), # red
        ego_successor_predecessor_color: T_ColorTuple = (0.7, 0.5, 0.1, 1.0), # orange
        ego_predecessor_successor_color: T_ColorTuple = (0.4, 0.4, 0.4, 1.0), # grey # (0.9, 0.9, 0.0, 1.0), # yellow
        fill_color: Optional[T_ColorTuple] = None,
        fill_resolution: int = 100,
        fill_offset: float = 0.05,
        from_graph: bool = False,
        graph_lanelet_id_attr: str = "id",
        graph_left_vertices_attr: str = "left_vertices",
        graph_center_vertices_attr: str = "left_vertices",
        graph_right_vertices_attr: str = "right_vertices",
        render_id: bool = False,
        truncate_id: bool = False,
        font_size: int = 6,
        label_color: T_ColorTuple = (255, 255, 255, 255),
        persistent: bool = True

    ) -> None:
        self._lanelet_linewidth = lanelet_linewidth
        self._lanelet_color = lanelet_color
        self._randomize_lanelet_color = randomize_lanelet_color
        self._enable_ego_rendering = enable_ego_rendering
        self._ego_outside_map_color = ego_outside_map_color
        self._ego_route_color = ego_route_color
        self._ego_lanelet_color = ego_lanelet_color
        self._ego_conflict_color = ego_conflict_color
        self._ego_successor_predecessor_color = ego_successor_predecessor_color
        self._ego_predecessor_successor_color = ego_predecessor_successor_color
        self._fill_color=fill_color
        self._fill_resolution=fill_resolution
        self._fill_offset=fill_offset
        self._from_graph = from_graph
        self._graph_lanelet_id_attr = graph_lanelet_id_attr
        self._graph_left_vertices_attr = graph_left_vertices_attr
        self._graph_center_vertices_attr = graph_center_vertices_attr
        self._graph_right_vertices_attr = graph_right_vertices_attr
        self._render_id = render_id
        self._truncate_id = truncate_id
        self._font_size = font_size
        self._label_color = label_color
        rng = np.random.default_rng(seed=0)
        self._rng_cache = CachedRNG(rng.random) if self._randomize_lanelet_color else None
        self._persistent = persistent
        self._last_drawn_scenario_id: Dict[Viewer2D, Optional[str]] = {}
        self._last_drawn_scenario_time_step: Dict[Viewer2D, int] = {}

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if viewer not in self._last_drawn_scenario_id:
            self._last_drawn_scenario_id[viewer] = None
            self._last_drawn_scenario_time_step[viewer] = params.time_step

        assert params.scenario is not None
        if self._persistent and (self._last_drawn_scenario_id[viewer] == params.scenario.scenario_id and (params.time_step == 1 + self._last_drawn_scenario_time_step[viewer] or params.time_step == self._last_drawn_scenario_time_step[viewer])):
            self._last_drawn_scenario_time_step[viewer] = params.time_step
            return
        self._last_drawn_scenario_id[viewer] = params.scenario.scenario_id
        self._last_drawn_scenario_time_step[viewer] = params.time_step

        if params.render_kwargs is not None:
            self._lanelet_linewidth = params.render_kwargs.pop('lanelet_linewidth', self._lanelet_linewidth)
        if self._lanelet_linewidth is None:
            if viewer.theme == ColorTheme.DARK:
                self._lanelet_linewidth = 0.64
            else:
                self._lanelet_linewidth = 0.30

        if self._from_graph:
            self._render_from_graph(viewer, params)
        else:
            self._render_from_scenario(viewer, params)

    def _render_from_scenario(self, viewer: Viewer2D, params: RenderParams):
        render_jobs: List[Tuple[Lanelet, T_ColorTuple, int]] = []
        has_ego_lanelet = params.ego_vehicle_simulation is not None and len(params.ego_vehicle_simulation.current_lanelet_ids) > 0

        if params.data is not None:
            included_lanelets_data = set(params.data.l.id.squeeze(1).tolist())
        else:
            included_lanelets_data = set()

        for lanelet in params.scenario.lanelet_network.lanelets:
            lanelet_render_zorder: int = 0
            color: Optional[T_ColorTuple] = None
            if params.data is not None and lanelet.lanelet_id not in included_lanelets_data:
                color = self._ego_outside_map_color
                lanelet_render_zorder = 0
            elif self._randomize_lanelet_color:
                color = self._rng_cache(
                    key=lanelet.lanelet_id,
                    n=3
                )
            elif has_ego_lanelet and self._enable_ego_rendering:
                ego_lanelet_id = params.ego_vehicle_simulation.current_lanelet_ids[0]
                if lanelet.lanelet_id == ego_lanelet_id:
                    color = self._ego_lanelet_color
                    lanelet_render_zorder = 3
                elif params.ego_vehicle is not None and params.ego_vehicle.ego_route is not None and lanelet.lanelet_id in params.ego_vehicle.ego_route.lanelet_id_route:
                    color = self._ego_route_color
                    lanelet_render_zorder = 1
                else:
                    inc_edge_types = {u: e['lanelet_edge_type'] for u, v, e in params.ego_vehicle_simulation.simulation.lanelet_graph.in_edges(ego_lanelet_id, data=True)}
                    lanelet_edge_type = inc_edge_types.get(lanelet.lanelet_id, None)
                    if lanelet_edge_type in {LaneletEdgeType.CONFLICTING, LaneletEdgeType.CONFLICT_LINK}:
                        color = self._ego_conflict_color
                        lanelet_render_zorder = 2
                    elif lanelet_edge_type == LaneletEdgeType.DIVERGING:
                        color = self._ego_predecessor_successor_color
                        lanelet_render_zorder = 2
                    elif lanelet_edge_type == LaneletEdgeType.MERGING:
                        color = self._ego_successor_predecessor_color
                        lanelet_render_zorder = 2
            if color is None:
                color = self._lanelet_color

            render_jobs.append((lanelet, color, lanelet_render_zorder))

        if has_ego_lanelet and self._enable_ego_rendering:
            render_jobs = sorted(render_jobs, key=lambda x: x[-1])

        for lanelet, color, _ in render_jobs:
            draw_lanelet(
                viewer=viewer,
                left_vertices=lanelet.left_vertices,
                center_vertices=lanelet.center_vertices,
                right_vertices=lanelet.right_vertices,
                color=color,
                linewidth=self._lanelet_linewidth,
                font_size=self._font_size,
                label=str(lanelet.lanelet_id) if self._render_id else None,
                fill_color=self._fill_color,
                fill_resolution=self._fill_resolution,
                fill_offset=self._fill_offset,
                end_marker=not lanelet.successor,
                persistent=self._persistent
            )

    def _render_from_graph(self, viewer: Viewer2D, params: RenderParams) -> None:
        if params.data is None:
            return

        left_vertices = params.data.l[self._graph_left_vertices_attr]
        center_vertices = params.data.l[self._graph_center_vertices_attr]
        right_vertices = params.data.l[self._graph_right_vertices_attr]
        mask: Optional[Tensor] = None
        if isinstance(left_vertices, Tensor) and left_vertices.dim() == 2:
            left_vertices = left_vertices.view(params.data.l.num_nodes, -1, 2)
            center_vertices = center_vertices.view(params.data.l.num_nodes, -1, 2)
            right_vertices = right_vertices.view(params.data.l.num_nodes, -1, 2)
            mask = ~((torch.isclose(left_vertices, left_vertices.new_zeros(left_vertices.shape))) |
                (torch.isclose(right_vertices, right_vertices.new_zeros(right_vertices.shape)))).all(dim=2)

        render_jobs: List[Tuple[int, int, bool, int]] = []
        has_ego_lanelet = params.ego_vehicle is not None
        if has_ego_lanelet:
            ego_vehicle_idx = torch.where(params.data.vehicle.is_ego_mask)[0][0].item()
            ego_lanelet_assignments = torch.where(params.data.v2l.edge_index[0, :] == ego_vehicle_idx)[0]
            if len(ego_lanelet_assignments) > 0:
                ego_lanelet_edge_idx = ego_lanelet_assignments[0].item()
                ego_lanelet_idx = params.data.v2l.edge_index[1, ego_lanelet_edge_idx].item()
            else:
                ego_vehicle_idx, ego_lanelet_idx = None, None
                has_ego_lanelet = False
        else:
            ego_vehicle_idx, ego_lanelet_idx = None, None

        for idx, lanelet_id in enumerate(params.data.l[self._graph_lanelet_id_attr]):
            if self._enable_ego_rendering:
                is_ego_lanelet = has_ego_lanelet and idx == ego_lanelet_idx
                priority = int(is_ego_lanelet)
            else:
                is_ego_lanelet, priority = False, 0

            render_jobs.append((idx, lanelet_id.item(), is_ego_lanelet, priority))

        if has_ego_lanelet and self._enable_ego_rendering:
            render_jobs = sorted(render_jobs, key=lambda x: x[-1])

        for idx, lanelet_id, is_ego_lanelet, priority in render_jobs:
            if self._randomize_lanelet_color:
                color = self._rng_cache(
                    key=lanelet_id,
                    n=3
                )
            elif is_ego_lanelet:
                color = self._ego_lanelet_color
            else:
                color = self._lanelet_color

            left_vertex = left_vertices[idx]
            center_vertex = center_vertices[idx]
            right_vertex = right_vertices[idx]
            if mask is not None:
                left_vertex = left_vertex[mask[idx]]
                center_vertex = center_vertex[mask[idx]]
                right_vertex = right_vertex[mask[idx]]

            draw_lanelet(
                viewer=viewer,
                left_vertices=left_vertex,
                center_vertices=center_vertex,
                right_vertices=right_vertex,
                color=color,
                linewidth=self._lanelet_linewidth,
                font_size=self._font_size,
                label=str(lanelet_id) if self._render_id else None,
                fill_color=self._fill_color,
                fill_resolution=self._fill_resolution,
                fill_offset=self._fill_offset,
                persistent=self._persistent
            )


def draw_lanelet(
    viewer: Viewer2D,
    left_vertices: Union[np.ndarray, ContinuousPolyline],
    center_vertices: Union[np.ndarray, ContinuousPolyline],
    right_vertices: Union[np.ndarray, ContinuousPolyline],
    color: T_ColorTuple,
    linewidth: float,
    label: Optional[str] = None,
    font_size: int = 12,
    fill_resolution: int = 100,
    fill_offset: float = 0.05,
    fill_color: Optional[T_ColorTuple] = None,
    start_marker: bool = True,
    end_marker: bool = True,
    persistent: bool = False,
    resample_interval: Optional[Union[float, int]] = None
) -> None:
    if left_vertices.shape[0] == 0:
        return

    left_vertices_arr = left_vertices.waypoints if isinstance(left_vertices, ContinuousPolyline) else left_vertices
    center_vertices_arr = center_vertices.waypoints if isinstance(center_vertices, ContinuousPolyline) else center_vertices
    right_vertices_arr = right_vertices.waypoints if isinstance(right_vertices, ContinuousPolyline) else right_vertices

    if resample_interval is not None:
        left_vertices_arr = np.array(resample_polyline(left_vertices_arr, resample_interval))
        center_vertices_arr = np.array(resample_polyline(center_vertices_arr, resample_interval))
        right_vertices_arr = np.array(resample_polyline(right_vertices_arr, resample_interval))

    if start_marker:
        viewer.draw_line(
            left_vertices_arr[0],
            right_vertices_arr[0],
            linewidth=linewidth,
            color=color,
            persistent=persistent
        )
    if end_marker:
        viewer.draw_line(
            left_vertices_arr[-1],
            right_vertices_arr[-1],
            linewidth=linewidth,
            color=color,
            persistent=persistent
        )
    viewer.draw_polyline(
        left_vertices_arr,
        linewidth=linewidth,
        color=color,
        persistent=persistent
    )
    viewer.draw_polyline(
        right_vertices_arr,
        linewidth=linewidth,
        color=color,
        persistent=persistent
    )

    if fill_color is not None:
        left_polyline = left_vertices if isinstance(left_vertices, ContinuousPolyline) else ContinuousPolyline(left_vertices_arr)
        center_polyline = center_vertices if isinstance(center_vertices, ContinuousPolyline) else ContinuousPolyline(center_vertices_arr)
        right_polyline = right_vertices if isinstance(right_vertices, ContinuousPolyline) else ContinuousPolyline(right_vertices_arr)

        path_length = center_polyline.length

        for t in range(fill_resolution):
            arclength = center_polyline.length*t/(fill_resolution - 1)
            pos = center_polyline(arclength)

            length = path_length/fill_resolution + fill_offset
            top_arclength = arclength + length
            bottom_arclength = arclength
            width = left_polyline.get_lateral_distance(right_polyline(arclength)) - 0.1
            top_left = center_polyline.lateral_translate_point(top_arclength, -width/2) - pos
            top_right = center_polyline.lateral_translate_point(top_arclength, width/2) - pos
            bottom_left = center_polyline.lateral_translate_point(bottom_arclength, -width/2) - pos
            bottom_right = center_polyline.lateral_translate_point(bottom_arclength, width/2) - pos

            vertices = np.array([
                bottom_left, top_left,
                top_right, bottom_right,
                bottom_left
            ])

            viewer.draw_shape(
                vertices=vertices,
                position=pos,
                filled=True,
                color=fill_color,
                linewidth=0,
                border=False,
                persistent=persistent
            )

    if label is not None:
        label = str(label)
        viewer.draw_label(
            #x=float(left_vertices[0, 0] + left_vertices[-1, 0] + right_vertices[0, 0] + right_vertices[-1, 0]) / 4,
            #y=float(left_vertices[0, 1] + left_vertices[-1, 1] + right_vertices[0, 1] + right_vertices[-1, 1]) / 4,
            x=float(left_vertices_arr[0, 0] + right_vertices_arr[0, 0]) / 2,
            y=float(left_vertices_arr[0, 1] + right_vertices_arr[0, 1]) / 2,
            label=label,
            color=(255, 255, 255, 255),
            font_size=font_size,
            persistent=persistent
        )
