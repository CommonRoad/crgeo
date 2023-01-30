from dataclasses import dataclass
from typing import Optional
import warnings
from crgeo.dataset.commonroad_data_temporal import CommonRoadDataTemporal

from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin
from crgeo.rendering.types import RenderParams
from crgeo.rendering.color_utils import T_ColorTuple
from crgeo.rendering.viewer.viewer_2d import Viewer2D


@dataclass
class RenderVehicleToLaneletEdgesStyle:
    edge_arc: float = 0.0
    edge_linewidth: float = 0.85
    edge_color_other_connection: T_ColorTuple = (0.9, 0.1, 0.1, 0.45)
    edge_color_ego_connection: T_ColorTuple = (0.0, 1.0, 0.0, 0.7)
    render_id=False
    label_color=(255, 255, 255, 255)
    font_size=4
    right_align_laterally: bool = True
    dashed: bool = False
    draw_ego_incoming_route: bool = False
    temporal_alpha_multiplier: float = 0.7
    render_temporal: bool = True


class RenderVehicleToLaneletEdgesPlugin(BaseRendererPlugin):
    def __init__(
        self,
        **kwargs
    ) -> None:
        self.style = RenderVehicleToLaneletEdgesStyle(**kwargs)


    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return

        is_temporal = isinstance(params.data, CommonRoadDataTemporal)
        edge_index = params.data.v2l.edge_index
        vehicle_pos = params.data.v.pos
        is_ego_mask = params.data.v.is_ego_mask

        if self.style.right_align_laterally:
            lanelet_pos_vec = params.data.l.right_pos
        else:
            lanelet_pos_vec = params.data.l.center_pos

        # lanelet_pos_vec = [] # TODO
        # for lidx, lid in enumerate(params.data.l.id):
        #     cl = params.ego_vehicle_simulation.simulation.get_lanelet_center_polyline(lid.item())
        #     pos_i = cl.lateral_translate_point(cl.length/2, lateral_distance=3.0)
        #     lanelet_pos_vec.append(pos_i)

        for idx in range(edge_index.shape[1]):
            # Replace idx here with one from indices
            vehicle_idx, lanelet_idx = params.data.v2l.edge_index[:, idx]
            vehicle_pos = params.data.v.pos[vehicle_idx].numpy()
            try:
                lanelet_pos = lanelet_pos_vec[lanelet_idx].numpy()
            except IndexError as e:
                warnings.warn(f"Could not render v2l edge to invalid lanelet index: {repr(e)}")
                continue
            ego_connection = bool(is_ego_mask[vehicle_idx].item())

            color = self.style.edge_color_ego_connection if ego_connection else self.style.edge_color_other_connection

            if is_temporal:
                timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[vehicle_idx].item()
                alpha_multiplier = self.style.temporal_alpha_multiplier**timestep
                color = color[:3] + (color[3]*alpha_multiplier,)
                if timestep != 0 and not self.style.render_temporal:
                    continue

            if params.data.v2l.arclength_rel[idx].item() > 0.5:
                arc = -self.style.edge_arc
            else:
                arc = self.style.edge_arc

            if self.style.dashed:
                viewer.draw_dashed_line(
                    vehicle_pos,
                    lanelet_pos,
                    linewidth=self.style.edge_linewidth,
                    color=color,
                    spacing=1.5,
                    arc=arc
                )
            else:
                viewer.draw_line(
                    vehicle_pos,
                    lanelet_pos,
                    linewidth=self.style.edge_linewidth,
                    color=color,
                    arc=arc
                )
            if self.style.render_id:
                viewer.draw_label(
                    x=(abs(vehicle_pos[0]) + abs(lanelet_pos[0])) / 2,
                    y=(vehicle_pos[1] + lanelet_pos[1]) / 2,
                    label=str(int(params.data.v.id[vehicle_idx].numpy()[0])) + ":" + str(int(params.data.l.id[lanelet_idx].numpy()[0])),
                    color=self.style.label_color,
                    font_size=self.style.font_size
                )

        if self.style.draw_ego_incoming_route:
            for lanelet_idx, lanelet_id in enumerate(params.data.l.id):
                if lanelet_id.item() in params.ego_vehicle_simulation.ego_route.lanelet_id_route[:4]:
                    viewer.draw_dashed_line(
                        lanelet_pos_vec[lanelet_idx].numpy(),
                        params.ego_vehicle.state.position,
                        linewidth=self.style.edge_linewidth,
                        color=self.style.edge_color_ego_connection,
                        arc=self.style.edge_arc
                    )
