import logging
from dataclasses import dataclass

from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer

logger = logging.getLogger(__name__)


@dataclass
class RenderVehicleToLaneletEdgesPlugin(BaseRenderPlugin):
    edge_arc: float = 0.0
    edge_linewidth: float = 0.3
    edge_color_other_connection: Color = Color((0.9, 0.1, 0.1, 0.45))
    edge_color_ego_connection: Color = Color((0.0, 1.0, 0.0, 0.7))
    render_id = False
    label_color = Color("white")
    font_size = 4
    right_align_laterally: bool = True
    dashed: bool = False
    draw_ego_incoming_route: bool = False
    temporal_alpha_multiplier: float = 0.7
    render_temporal: bool = True

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.data is None:
            return

        is_temporal = isinstance(params.data, CommonRoadDataTemporal)
        edge_index = params.data.v2l.edge_index
        is_ego_mask = params.data.v.is_ego_mask

        if self.right_align_laterally:
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
                logger.warning(f"Could not render v2l edge to invalid lanelet index: {repr(e)}")
                continue
            ego_connection = bool(is_ego_mask[vehicle_idx].item())

            color = self.edge_color_ego_connection if ego_connection else self.edge_color_other_connection

            if is_temporal:
                timestep = params.data.vehicle.batch.max().item() - params.data.vehicle.batch[vehicle_idx].item()
                alpha_multiplier = self.temporal_alpha_multiplier ** timestep
                color = color.with_alpha(alpha=alpha_multiplier)
                if timestep != 0 and not self.render_temporal:
                    continue

            if params.data.v2l.arclength_rel[idx].item() > 0.5:
                arc = -self.edge_arc
            else:
                arc = self.edge_arc

            if self.dashed:
                viewer.draw_dashed_line(
                    creator=self.__class__.__name__,
                    start=vehicle_pos,
                    end=lanelet_pos,
                    line_width=self.edge_linewidth,
                    color=color,
                    spacing=1.5,
                    arc=arc
                )
            else:
                viewer.draw_line(
                    creator=self.__class__.__name__,
                    start=vehicle_pos,
                    end=lanelet_pos,
                    line_width=self.edge_linewidth,
                    color=color,
                    arc=arc
                )

            if self.render_id:
                viewer.draw_label(
                    creator=self.__class__.__name__,
                    text=str(int(params.data.v.id[vehicle_idx].numpy()[0])) + ":" + str(int(params.data.l.id[lanelet_idx].numpy()[0])),
                    color=self.label_color,
                    x=(abs(vehicle_pos[0]) + abs(lanelet_pos[0])) / 2,
                    y=(vehicle_pos[1] + lanelet_pos[1]) / 2,
                    font_size=self.font_size
                )

        if self.draw_ego_incoming_route:
            for lanelet_idx, lanelet_id in enumerate(params.data.l.id):
                if lanelet_id.item() in params.ego_vehicle_simulation.ego_route.lanelet_id_route[:4]:
                    viewer.draw_dashed_line(
                        creator=self.__class__.__name__,
                        start=lanelet_pos_vec[lanelet_idx].numpy(),
                        end=params.ego_vehicle.state.position,
                        line_width=self.edge_linewidth,
                        color=self.edge_color_ego_connection,
                        arc=self.edge_arc
                    )
