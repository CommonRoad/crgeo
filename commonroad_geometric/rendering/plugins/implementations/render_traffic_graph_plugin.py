from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType, LaneletEdgeTypeColorMap
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderTrafficGraphPlugin(BaseRenderPlugin):
    edge_linewidth: float = 0.3
    edge_arc: float = 0.0
    edge_color_other_connection: Color = Color((0.0, 0.9, 0.0, 0.4))
    edge_color_ego_connection: Color = Color((0.1, 0.9, 0.1, 0.5))
    edge_color_temporal: Color = Color((0.0, 0.0, 0.9, 0.4))
    render_id: bool = False
    render_temporal: bool = True
    label_color: Color = Color("white")
    font_size: float = 4
    vehicle_pos_attr: str = "pos"
    is_ego_mask_attr: Optional[str] = "is_ego_mask"
    node_radius: Optional[float] = None
    node_linewidth: float = 0.01
    node_linecolor: Color = Color("black")
    node_fillcolor: Color = Color("green")
    node_fillcolor_ego: Color = Color("green")
    draw_ego: bool = False
    temporal_alpha_multiplier: float = 0.7
    temporal_edge_arc: float = 0.05

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return
        self._render_v2v(viewer, params)
        if self.render_temporal:
            self._render_vtv(viewer, params)
        self._render_nodes(viewer, params)

    def _render_vtv(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ):
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)
        if not is_temporal:
            return

        edge_index = params.data.vtv.edge_index
        vehicle_pos = params.data.vehicle[self.vehicle_pos_attr]

        if not edge_index.numel() or not vehicle_pos.numel():
            return

        for edge in edge_index.T:
            from_vehicle_idx = edge[0].item()
            to_vehicle_idx = edge[1].item()

            from_vehicle_pos = vehicle_pos[from_vehicle_idx].numpy(force=True)
            to_vehicle_pos = vehicle_pos[to_vehicle_idx].numpy(force=True)

            color = self.edge_color_temporal

            timestep_from = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[from_vehicle_idx].item()
            timestep_to = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[to_vehicle_idx].item()

            alpha_multiplier = self.temporal_alpha_multiplier ** timestep_from
            color = color.with_alpha(alpha=alpha_multiplier)

            if abs(timestep_to - timestep_from) <= 1:
                arc = 0.0
            else:
                arc = self.temporal_edge_arc if timestep_to % 2 == 0 else -self.temporal_edge_arc

            viewer.draw_line(
                creator=self.__class__.__name__,
                start=from_vehicle_pos,
                end=to_vehicle_pos,
                color=color,
                line_width=self.edge_linewidth,
                # arc=arc
            )

    def _render_nodes(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ):
        vehicle_pos = params.data.vehicle[self.vehicle_pos_attr]
        is_ego_mask: Optional[Tensor] = None
        if self.is_ego_mask_attr is not None:
            is_ego_mask = params.data.vehicle[self.is_ego_mask_attr]
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)

        if self.node_radius is not None:
            for idx, pos in enumerate(vehicle_pos):
                if is_ego_mask[idx] and not self.draw_ego:
                    continue
                color = self.node_fillcolor_ego if is_ego_mask[idx] else self.node_fillcolor
                linecolor = self.node_linecolor

                if is_temporal:
                    timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[idx].item()
                    alpha_multiplier = self.temporal_alpha_multiplier ** timestep
                    color = color.with_alpha(alpha=alpha_multiplier)
                    linecolor = linecolor.with_alpha(alpha=alpha_multiplier)
                    if timestep != 0 and not self.render_temporal:
                        continue

                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=pos,
                    radius=self.node_radius,
                    fill_color=color,
                    border_color=linecolor,
                    line_width=self.node_linewidth,
                )

    def _render_v2v(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ):
        edge_index = params.data.vehicle_to_vehicle.edge_index
        vehicle_pos = params.data.vehicle[self.vehicle_pos_attr]

        if isinstance(vehicle_pos, torch.Tensor):
            vehicle_pos = vehicle_pos.numpy()

        is_ego_mask: Optional[Tensor] = None
        if self.is_ego_mask_attr is not None:
            is_ego_mask = params.data.vehicle[self.is_ego_mask_attr]
        is_temporal = isinstance(params.data, CommonRoadDataTemporal)

        for edge in edge_index.T:
            from_vehicle_idx = edge[0].item()
            to_vehicle_idx = edge[1].item()

            from_vehicle_pos = vehicle_pos[from_vehicle_idx]
            to_vehicle_pos = vehicle_pos[to_vehicle_idx]

            ego_connection = is_ego_mask is not None and bool(is_ego_mask[to_vehicle_idx].item())
            if ego_connection:
                # v2l = params.data.v2l.edge_index
                # l2l = params.data.l2l.edge_index
                # from_lanelet_indices = torch.where(v2l[0, :] == from_vehicle_idx)[0]
                # to_lanelet_indices = torch.where(v2l[0, :] == to_vehicle_idx)[0]
                # l2l_dominant_edge_type = -1
                # for from_lanelet_idx_th in from_lanelet_indices:
                #     for to_lanelet_idx_th in to_lanelet_indices:
                #         from_lanelet_idx = v2l[1, from_lanelet_idx_th.item()].item()
                #         to_lanelet_idx = v2l[1, to_lanelet_idx_th.item()].item()
                #         if params.data.l.id[to_lanelet_idx].item() not in params.ego_vehicle.ego_route.lanelet_id_route:
                #             continue
                #         if from_lanelet_idx == to_lanelet_idx:
                #             lanelet_edge_type = LaneletEdgeType.SUCCESSOR.value
                #         else:
                #             matches = torch.where((l2l[0] == from_lanelet_idx) & (l2l[1] == to_lanelet_idx))
                #             lanelet_edge_match = matches[0]
                #             if len(lanelet_edge_match) > 0:
                #                 assert len(lanelet_edge_match) == 1
                #                 lanelet_edge_idx = lanelet_edge_match.item()
                #                 lanelet_edge_type = params.data.l2l.type[lanelet_edge_idx].item()
                #             else:
                #                 lanelet_edge_type = -1
                #         if lanelet_edge_type > l2l_dominant_edge_type:
                #             l2l_dominant_edge_type = lanelet_edge_type

                # if l2l_dominant_edge_type > -1:
                #     color = LaneletEdgeTypeColorMap[LaneletEdgeType(l2l_dominant_edge_type)]
                # else:
                color = self.edge_color_ego_connection
            else:
                color = self.edge_color_other_connection

            if is_temporal:
                timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[from_vehicle_idx].item()
                alpha_multiplier = self.temporal_alpha_multiplier ** timestep
                color = color.with_alpha(alpha=alpha_multiplier)
                if timestep != 0 and not self.render_temporal:
                    continue

            viewer.draw_line(
                creator=self.__class__.__name__,
                start=from_vehicle_pos,
                end=to_vehicle_pos,
                line_width=self.edge_linewidth,
                color=color,
                arc=self.edge_arc
            )

            if self.render_id:
                from_vehicle_id = int(params.data.v.id[from_vehicle_idx].numpy()[0])
                to_vehicle_id = int(params.data.v.id[to_vehicle_idx].numpy()[0])
                viewer.draw_label(
                    creator=self.__class__.__name__,
                    text=f"{from_vehicle_id}:{to_vehicle_id}",
                    color=self.label_color,
                    font_size=self.font_size,
                    x=(abs(from_vehicle_pos[0]) + abs(to_vehicle_pos[0])) / 2,
                    y=(from_vehicle_pos[1] + to_vehicle_pos[1]) / 2,
                )
