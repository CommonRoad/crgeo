from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from commonroad.scenario.lanelet import Lanelet

from commonroad_geometric.common.io_extensions.obstacle import get_state_list
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import draw_lanelet
from commonroad_geometric.rendering.plugins.implementations.render_traffic_flow_lanelet_network_plugin import RenderTrafficFlowLaneletNetworkPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


from commonroad_geometric.rendering.plugins.helpers import (
    create_inner_vehicle_polylines_from_lanelet_polylines,
    lanelet_is_valid,
    resample_lanelet
)

def calculate_extrapolation_distance(
    v_features, obstacle_idx: int, dt: float, exploration_length: int
) -> Tuple[float, float, float]:
    vel_idx = v_features.column_indices["velocity"][0]  # 1 is vel_y which is always 0
    acc_idx = v_features.column_indices["acceleration"][
        0
    ]  # 1 is acc_y which is always 0
    current_v = v_features.x[obstacle_idx, vel_idx]
    current_a = v_features.x[obstacle_idx, acc_idx]

    exploration_time = dt * exploration_length
    extrapolated_distance = current_v * exploration_time + 0.5 * current_a * (
        exploration_time**2
    )

    length_idx = v_features.column_indices["length"][0]
    length_offset = v_features.x[obstacle_idx, length_idx] / 2
    width_idx = v_features.column_indices["width"][0]
    width_offset = v_features.x[obstacle_idx, width_idx] / 2
    return extrapolated_distance.item(), length_offset.item(), width_offset.item()


def _collect_best_lanelets(
    params: RenderParams,
    potential_lanelet_ids: List[int],
    obstacle_idx: int,
) -> List[Lanelet]:
    # Assume that vehicle is driving in the right direction and collect matching lanelets.
    (
        viable_lanelets,
        heading_errors,
        _,
    ) = RenderTrafficFlowLaneletNetworkPlugin._collect_viable_lanelets(
        params.data, params.scenario, potential_lanelet_ids, obstacle_idx
    )

    if len(viable_lanelets) <= 1:
        return viable_lanelets

    cut_idx = 1
    for heading_error in heading_errors[1:]:
        if heading_error == heading_errors[0]:
            cut_idx += 1

    return viable_lanelets[:cut_idx]



@dataclass
class RenderObstacleExtrapolation(BaseRenderObstaclePlugin):
    extrapolation_length: int = 5
    extrapolation_color: Optional[Color] = Color('red')

    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ):
        lanelets = params.simulation.lanelet_network._lanelets
        for obstacle_idx, obstacle_id in enumerate(params.data.v.id.H[0]):
            obstacle_id = obstacle_id.item()
            # we don't render extrapolation for the ego vehicle.
            if params.data.v.is_ego_mask[obstacle_idx]:
                continue

            (
                extrapolation_distance,
                length_offset,
                width_offset,
            ) = calculate_extrapolation_distance(
                params.data.v,
                obstacle_idx,
                params.scenario.dt,
                self.extrapolation_length,
            )

            # We cannot create a Lanelet that short. It's irrelevant.
            if extrapolation_distance <= 0.04:
                continue

            potential_lanelet_ids = (
                params.simulation.obstacle_id_to_lanelet_id[obstacle_id]
            )

            # NOTE: Exploration is only rendered for surrounding vehicles. As their driving
            # is perfect, we only need the best lanelets and not all viable lanelets.
            best_lanelets = _collect_best_lanelets(
                params, potential_lanelet_ids, obstacle_idx
            )


            final_lanelets: List[Lanelet] = []
            for lanelet in best_lanelets:
                result = self.extrapolate_(
                    lanelets,
                    lanelet,
                    extrapolation_distance + length_offset,
                    width_offset,
                )

                if result is None:
                    continue

                if isinstance(result, list):
                    final_lanelets += result

            for lanelet in final_lanelets:
                draw_lanelet(
                    creator=self.__class__.__name__,
                    viewer=viewer,
                    left_vertices=lanelet.left_vertices,
                    center_vertices=lanelet.center_vertices,
                    right_vertices=lanelet.right_vertices,
                    color=self.extrapolation_color,
                    fill_color=self.extrapolation_color,
                    start_marker=True,
                    end_marker=True,
                    is_persistent=False,
                    linewidth=0.0,
                )

    def extrapolate_(
        self,
        lanelets: Dict[int, Lanelet],
        lanelet: Lanelet,
        extrapolated_distance: float,
        width_offset: float,
    ) -> List[Lanelet] | None:
        ret_lanelets: List[Lanelet] = []

        border_indices = np.where(lanelet.distance >= extrapolated_distance)[0]
        if len(border_indices) == 0:
            # Extrapolation extends over lanelet boundary.
            cut_idx = len(lanelet.center_vertices)
        else:
            # Lanelet is long enough to contain extrapolation.
            cut_idx = border_indices[0] + 1

        (
            left_vertices,
            right_vertices,
        ) = create_inner_vehicle_polylines_from_lanelet_polylines(
            lanelet.left_vertices[:cut_idx],
            lanelet.center_vertices[:cut_idx],
            lanelet.right_vertices[:cut_idx],
            width_offset,
        )

        # left_vertices = lanelet.left_vertices[:cut_idx]
        center_vertices = lanelet.center_vertices[:cut_idx]
        # right_vertices = lanelet.right_vertices[:cut_idx]

        if lanelet_is_valid(left_vertices, center_vertices, right_vertices):
            ret_lanelet = Lanelet(
                lanelet_id=1337,
                left_vertices=left_vertices,
                center_vertices=center_vertices,
                right_vertices=right_vertices,
                successor=lanelet.successor,
            )

            ret_lanelets.append(ret_lanelet)

            remaining_distance = extrapolated_distance - ret_lanelet.distance[-1]
            if remaining_distance > 0.04:
                for succ in ret_lanelet.successor:
                    resample_lanelet(lanelets[succ], 0.04)
                    result = self.extrapolate_(
                        lanelets,
                        lanelets[succ],
                        remaining_distance,
                        width_offset,
                    )
                    if result is not None:
                        ret_lanelets += result
            return ret_lanelets
