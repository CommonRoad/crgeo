from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import math
import logging

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import resample_polyline
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal

from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.helpers import (
    attach_vehicle_to_lanelet,
    get_heading_error,
    get_v2l_edge_idx,
)
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import draw_lanelet
from commonroad_geometric.rendering.types import RenderParams


logger = logging.getLogger(__name__)


class RenderTrafficFlowLaneletNetworkPlugin(BaseRenderPlugin):
    def __init__(
        self,
        line_width: float = 0.1,
        fill_resolution: int = 20,
        fill_offset: float = 0.4,
        neighbor_lanelet_color: Color = Color((0.36, 0.7, 1.0, 0.6)),
    ) -> None:
        self._line_width = line_width
        self._fill_resolution = fill_resolution
        self._fill_offset = fill_offset
        self._fill_color = neighbor_lanelet_color

    def render(self, viewer: BaseViewer, params: RenderParams) -> None:
        self._render_from_scenario(viewer, params)

    @staticmethod
    def _collect_viable_lanelets(
        data: CommonRoadData | CommonRoadDataTemporal,
        scenario: Scenario,
        potential_lanelet_ids: List[int],
        ego_obstacle_idx: int,
    ) -> Tuple[List[Lanelet], List[float], List[float]]:
        ret_lanelets: List[Lanelet] = []
        ret_heading_errors: List[float] = []
        ret_arclength_abs: List[float] = []

        lanelets = scenario.lanelet_network._lanelets
        for lanelet_id in potential_lanelet_ids:
            # NOTE: Do NOT use lanelet_id_to_lanelet_idx here! EgoVehicleSimulation fails.
            l_features = torch.where(data.l.id == lanelet_id)[0]
            if l_features.numel() == 0:
                # This can happen in a EgoVehicleSimulation.
                continue

            lanelet_idx = l_features[0].item()
            try:
                v2l_edge_index = get_v2l_edge_idx(
                    lanelet_idx, ego_obstacle_idx, data.v2l["edge_index"]
                )
            except IndexError:
                continue
            heading_error = get_heading_error(data.v2l, v2l_edge_index)
            ret_heading_errors.append(heading_error)
            arclength_abs = data.v2l["arclength_abs"][v2l_edge_index].item()
            ret_arclength_abs.append(arclength_abs)

            # 43 degrees, 70% of BMW 320i max steering angle (1.06).
            if abs(heading_error) > 0.75:
                # Lanelet-vehicle orientation is too large and lanelet cannot be
                # considered as drivable lanelet. The others will be even larger.
                break

            new_lanelet = attach_vehicle_to_lanelet(
                data.v.pos[ego_obstacle_idx],
                lanelets[lanelet_id],
                arclength_abs,
            )

            # TODO: This might lead to an empty ret_lanelets eventhough there are matching lanelets!
            if new_lanelet is not None:
                ret_lanelets.append(new_lanelet)
        return ret_lanelets, ret_heading_errors, ret_arclength_abs

    def _collect_opposite_adjacent_lanelets(
        self,
        params: RenderParams,
        potential_lanelet_ids: List[int],
        heading_errors: List[float],
        arclength_abss: List[float],
        ego_position: np.ndarray,
    ) -> List[Lanelet]:
        ret_lanelets = []

        lanelets = params.scenario.lanelet_network._lanelets

        # Ideally, the lanelet is perfectly opposite (180° = math.pi).
        opposite_idx_arr = np.argsort(np.abs(heading_errors) - math.pi)
        for opposite_id_idx in opposite_idx_arr:
            if not (
                (math.pi - 0.75)
                <= np.abs(heading_errors[opposite_id_idx])
                <= (math.pi + 0.75)
            ):
                break

            opposite_id = potential_lanelet_ids[opposite_id_idx]
            wrong_direction_lanelet = lanelets[opposite_id]
            if (
                wrong_direction_lanelet.adj_left is not None
                and not wrong_direction_lanelet.adj_left_same_direction
            ):
                ret = attach_vehicle_to_lanelet(
                    ego_position,
                    lanelets[wrong_direction_lanelet.adj_left],
                    wrong_direction_lanelet.distance[-1]
                    - arclength_abss[opposite_id_idx],
                )
                if ret is not None:
                    ret_lanelets.append(ret)

            if (
                wrong_direction_lanelet.adj_right is not None
                and not wrong_direction_lanelet.adj_right_same_direction
            ):
                ret = attach_vehicle_to_lanelet(
                    ego_position,
                    lanelets[wrong_direction_lanelet.adj_right],
                    wrong_direction_lanelet.distance[-1]
                    - arclength_abss[opposite_id_idx],
                )
                if ret is not None:
                    ret_lanelets.append(ret)

        return ret_lanelets

    def _render_from_scenario(self, viewer: BaseViewer, params: RenderParams):
        lanelets = params.scenario.lanelet_network._lanelets

        ego_obstacle_idx = torch.where(params.data.v.is_ego_mask)[0][0].item()
        ego_obstacle_id = params.data.v.id[ego_obstacle_idx].item()

        potential_lanelet_ids = params.simulation.obstacle_id_to_lanelet_id[
            ego_obstacle_id
        ]

        # Assume that vehicle is driving in the right direction and collect matching lanelets.
        (
            viable_lanelets,
            heading_errors,
            arclength_abss,
        ) = RenderTrafficFlowLaneletNetworkPlugin._collect_viable_lanelets(
            params.data, params.scenario, potential_lanelet_ids, ego_obstacle_idx
        )

        # Vehicle is very likely driving in the opposite direction.
        if len(viable_lanelets) == 0:
            viable_lanelets = self._collect_opposite_adjacent_lanelets(
                params,
                potential_lanelet_ids,
                heading_errors,
                arclength_abss,
                params.data.v.pos[ego_obstacle_idx],
            )

        # We could go further and search for adjacent lanelets from predecessors of the
        # opposite lanelets but those cases a rare and 1) the model won't generalize to
        # them anyway, and 2) the agent won't be able to plan actions that far ahead.
        # Also, it could be that the vehicle is horizontal to the road.
        if len(viable_lanelets) == 0:
            return

        def get_recursive_successors(pivot: Lanelet, depth: int):
            def recursive_helper(lanelet: Lanelet, current_depth: int):
                nonlocal lanelets

                if current_depth == 0:
                    return []

                result = []
                for successor_id in lanelet.successor:
                    successor = lanelets[successor_id]
                    if successor:
                        result.append(successor)
                        result.extend(recursive_helper(successor, current_depth - 1))
                return result

            return recursive_helper(pivot, depth)

        final_lanelets: List[Lanelet] = []
        for lanelet in viable_lanelets:
            final_lanelets.append(lanelet)
            final_lanelets += get_recursive_successors(lanelet, 5)

        for lanelet in final_lanelets:
            draw_lanelet(
                creator=self.__class__.__name__,
                viewer=viewer,
                left_vertices=lanelet.left_vertices,
                center_vertices=lanelet.center_vertices,
                right_vertices=lanelet.right_vertices,
                color=self._fill_color,  # (0.3, 0.1, 1.0, 1.0)
                linewidth=self._line_width,
                fill_resolution=self._fill_resolution,
                fill_offset=self._fill_offset,
                fill_color=self._fill_color,
                end_marker=False,
                is_persistent=False,
            )
