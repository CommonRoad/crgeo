from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal

from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletEdgeTypeColorMap
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.color_utils import T_ColorTuple
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderTrafficGraphStyle:
    edge_linewidth: float = 0.85
    edge_arc: float = 0.0
    edge_color_other_connection: T_ColorTuple = (0.0, 0.9, 0.0, 0.4)
    edge_color_ego_connection: T_ColorTuple = (0.0, 0.0, 0.9, 0.35)
    edge_color_temporal: T_ColorTuple = (0.0, 0.0, 0.9, 0.4)
    render_id: bool = False
    render_temporal: bool = True
    label_color: T_ColorTuple = (255, 255, 255, 255)
    font_size: float = 4
    vehicle_pos_attr: str = "pos"
    is_ego_mask_attr: Optional[str] = "is_ego_mask"
    node_radius: Optional[float] = None
    node_linewidth: float = 0.01
    node_linecolor: T_ColorTuple = (0.0, 0.0, 0.0, 1.0)
    node_fillcolor: T_ColorTuple = (0.0, 1.0, 0.0, 1.0)
    node_fillcolor_ego: T_ColorTuple = (0.0, 1.0, 0.0, 1.0)
    draw_ego: bool = False
    temporal_alpha_multiplier: float = 0.7
    temporal_edge_arc: float = 0.05

class RenderTrafficGraphPlugin(BaseRendererPlugin):

    def __init__(
        self,
        style: Optional[RenderTrafficGraphStyle] = None,
        **kwargs
    ) -> None:
        self.style = style if style is not None else RenderTrafficGraphStyle(**kwargs)

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return
        self._render_v2v(viewer, params)
        if self.style.render_temporal:
            self._render_vtv(viewer, params)
        self._render_nodes(viewer, params)

    def _render_vtv(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ):
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)
        if not is_temporal:
            return

        edge_index = params.data.vtv.edge_index
        vehicle_pos = params.data.vehicle[self.style.vehicle_pos_attr]

        for edge in edge_index.T:
            from_vehicle_idx = edge[0].item()
            to_vehicle_idx = edge[1].item()

            from_vehicle_pos = vehicle_pos[from_vehicle_idx]
            to_vehicle_pos = vehicle_pos[to_vehicle_idx]

            color = self.style.edge_color_temporal

            timestep_from = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[from_vehicle_idx].item()
            timestep_to = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[to_vehicle_idx].item()

            alpha_multiplier = self.style.temporal_alpha_multiplier**timestep_from
            color = color[:3] + (color[3]*alpha_multiplier,)

            if abs(timestep_to - timestep_from) <= 1:
                arc = 0.0
            else:
                arc = self.style.temporal_edge_arc if timestep_to % 2 == 0 else -self.style.temporal_edge_arc

            viewer.draw_line(
                from_vehicle_pos,
                to_vehicle_pos,
                linewidth=self.style.edge_linewidth,
                color=color,
                arc=arc
            )

    def _render_nodes(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ):
        vehicle_pos = params.data.vehicle[self.style.vehicle_pos_attr]
        is_ego_mask: Optional[Tensor] = None
        if self.style.is_ego_mask_attr is not None:
            is_ego_mask = params.data.vehicle[self.style.is_ego_mask_attr]
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)

        if self.style.node_radius is not None:
            for idx, pos in enumerate(vehicle_pos):
                if is_ego_mask[idx] and not self.style.draw_ego:
                    continue
                color = self.style.node_fillcolor_ego if is_ego_mask[idx] else self.style.node_fillcolor
                linecolor = self.style.node_linecolor

                if is_temporal:
                    timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[idx].item()
                    alpha_multiplier = self.style.temporal_alpha_multiplier**timestep
                    color = color[:3] + (color[3]*alpha_multiplier,)
                    linecolor = linecolor[:3] + (linecolor[3]*alpha_multiplier,)
                    if timestep != 0 and not self.style.render_temporal:
                        continue

                viewer.draw_circle(
                    origin=pos,
                    radius=self.style.node_radius,
                    filled=True,
                    color=color,
                    linewidth=self.style.node_linewidth,
                    linecolor=linecolor
                )

    def _render_v2v(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ):
        edge_index = params.data.vehicle_to_vehicle.edge_index
        vehicle_pos = params.data.vehicle[self.style.vehicle_pos_attr]
        is_ego_mask: Optional[Tensor] = None
        if self.style.is_ego_mask_attr is not None:
            is_ego_mask = params.data.vehicle[self.style.is_ego_mask_attr]
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)

        for edge in edge_index.T:
            from_vehicle_idx = edge[0].item()
            to_vehicle_idx = edge[1].item()

            from_vehicle_pos = vehicle_pos[from_vehicle_idx]
            to_vehicle_pos = vehicle_pos[to_vehicle_idx]

            ego_connection = is_ego_mask is not None and bool(is_ego_mask[to_vehicle_idx].item())
            if ego_connection:
                v2l = params.data.v2l.edge_index
                l2l = params.data.l2l.edge_index
                from_lanelet_indices = torch.where(v2l[0, :] == from_vehicle_idx)[0]
                to_lanelet_indices = torch.where(v2l[0, :] == to_vehicle_idx)[0]
                l2l_dominant_edge_type = -1
                for from_lanelet_idx_th in from_lanelet_indices:
                    for to_lanelet_idx_th in to_lanelet_indices:
                        from_lanelet_idx = v2l[1, from_lanelet_idx_th.item()].item()
                        to_lanelet_idx = v2l[1, to_lanelet_idx_th.item()].item()
                        if params.data.l.id[to_lanelet_idx].item() not in params.ego_vehicle.ego_route.lanelet_id_route:
                            continue
                        if from_lanelet_idx == to_lanelet_idx:
                            lanelet_edge_type = LaneletEdgeType.SUCCESSOR.value
                        else:
                            lanelet_edge_match = torch.where((l2l[0] == from_lanelet_idx) & (l2l[1] == to_lanelet_idx))[0]
                            if len(lanelet_edge_match) > 0:
                                assert len(lanelet_edge_match) == 1
                                lanelet_edge_idx = lanelet_edge_match.item()
                                lanelet_edge_type = params.data.l2l.type[lanelet_edge_idx].item()
                            else:
                                lanelet_edge_type = -1
                        if lanelet_edge_type > l2l_dominant_edge_type:
                            l2l_dominant_edge_type = lanelet_edge_type
                            
                if l2l_dominant_edge_type > -1: 
                    color = LaneletEdgeTypeColorMap[LaneletEdgeType(l2l_dominant_edge_type)]
                else:
                    color = self.style.edge_color_ego_connection
            else:
                color = self.style.edge_color_other_connection

            if is_temporal:
                timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[from_vehicle_idx].item()
                alpha_multiplier = self.style.temporal_alpha_multiplier**timestep
                color = color[:3] + (color[3]*alpha_multiplier,)
                if timestep != 0 and not self.style.render_temporal:
                    continue

            viewer.draw_line(
                from_vehicle_pos,
                to_vehicle_pos,
                linewidth=self.style.edge_linewidth,
                color=color,
                arc=self.style.edge_arc
            )

            if self.style.render_id:
                viewer.draw_label(
                    x=(abs(from_vehicle_pos[0]) + abs(to_vehicle_pos[0])) / 2,
                    y=(from_vehicle_pos[1] + to_vehicle_pos[1]) / 2,
                    label=str(int(params.data.v.id[from_vehicle_idx].numpy()[0])) + ":" + str(int(params.data.v.id[to_vehicle_idx].numpy()[0])),
                    color=self.style.label_color,
                    font_size=self.style.font_size
                )
