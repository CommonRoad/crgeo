import logging
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer, T_Viewer
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import V2V_Feature

logger = logging.getLogger(__name__)


@dataclass
class RenderObstacleFlowPlugin(BaseRenderObstaclePlugin):
    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        ego_edge_mask = params.data.v2v.edge_index[1, :] == self.reference_vehicle_idx
        obstacle_indices = params.data.v2v.edge_index[0, ego_edge_mask].tolist()
        try:
            relative_velocity_th = params.data.v2v[V2V_Feature.RelativeVelocityEgo.value][ego_edge_mask]
        except IndexError:
            raise

        for edge_idx, obstacle_idx in enumerate(obstacle_indices):

            obstacle = params.simulation.current_obstacles[obstacle_idx]
            if self.skip_ego_id and obstacle.obstacle_id < 0:
                continue
            if obstacle.obstacle_id in self.ignore_obstacle_ids:
                continue

            obstacle_vertices = params.data.v.vertices[obstacle_idx].view(-1, 2).numpy()
            obstacle_position = params.data.v.pos[obstacle_idx].tolist()
            obstacle_orientation = params.data.v.orientation[obstacle_idx].item()

            relative_velocity = relative_velocity_th[edge_idx]
            r = np.clip(0.5 + relative_velocity[0].item() / 100, 0.0, 1.0)
            g = np.clip(0.5 + relative_velocity[1].item() / 100, 0.0, 1.0)
            b = 0.0
            color = Color((r, g, b))

            viewer.draw_2d_shape(
                creator=self.__class__.__name__,
                vertices=obstacle_vertices,
                fill_color=color,
                border_color=None,
                translation=obstacle_position,
                rotation=obstacle_orientation,
                line_width=0.0
            )
