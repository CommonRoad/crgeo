import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.color.theme import ColorTheme
from commonroad_geometric.rendering.plugins.obstacles.base_obstacle_render_plugin import BaseRenderObstaclePlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer, T_Viewer

logger = logging.getLogger(__name__)


@dataclass
class RenderObstaclePlugin(BaseRenderObstaclePlugin):
    from_graph: bool = False
    graph_vertices_attr: Optional[str] = "vertices"
    graph_position_attr: Optional[str] = "pos"
    graph_orientation_attr: Optional[str] = "orientation"
    vehicle_expansion: float = 1.0  # Scale factor for vehicle size

    def _expand_vertices(self, vertices: list) -> list:
        """
        Expand vertices using a simple scaling transformation around centroid
        """
        # Convert to numpy array for easier manipulation
        vertices_array = np.array(vertices)
        
        # Calculate centroid
        centroid = np.mean(vertices_array, axis=0)
        
        # Translate to origin, scale, then translate back
        expanded_vertices = (vertices_array - centroid) * self.vehicle_expansion + centroid
        
        return expanded_vertices

    def render(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        if self.obstacle_fill_color is None:
            if viewer.options.theme == ColorTheme.BRIGHT:
                self.obstacle_fill_color = Color((0.8, 0.8, 0.8, 1.0))
            else:
                self.obstacle_fill_color = Color((0.6, 1.0, 0.6, 0.0))
        if self.obstacle_color is None:
            if viewer.options.theme == ColorTheme.BRIGHT:
                self.obstacle_color = Color((0.2, 0.2, 0.2, 1.0))
            else:
                self.obstacle_color = Color((0.1, 0.8, 0.1, 1.0))
        if self.obstacle_line_width is None:
            if viewer.options.theme == ColorTheme.BRIGHT:
                self.obstacle_line_width = 0.5
            else:
                self.obstacle_line_width = 0.7

        if self.from_graph:
            self._render_from_graph(viewer, params)
        else:
            self._render_from_scenario(viewer, params)

    def _render_from_scenario(
        self,
        viewer: BaseViewer,
        params: RenderParams,
    ) -> None:
        for obstacle in params.simulation.current_obstacles:
            if self.skip_ego_id and obstacle.obstacle_id < 0:
                continue
            if obstacle.obstacle_id in self.ignore_obstacle_ids:
                continue

            state = state_at_time(obstacle, params.time_step)
            vertices = obstacle.obstacle_shape.vertices
            if self.vehicle_expansion != 1.0:
                vertices = self._expand_vertices(vertices)

            color = self.obstacle_color
            fill_color = self.obstacle_fill_color
            if self.randomize_color_from is not None:
                rng_cache = self.get_active_rng_cache(viewer, params)
                match self.randomize_color_from:
                    case "obstacle":
                        rbg_tuple: tuple[float, float, float] = rng_cache(
                            key=obstacle.obstacle_id,
                            n=3
                        )
                        color = Color(rbg_tuple)
                        fill_color = color * 0.7
                    case "lanelet" | "viewer" | _:
                        obstacle_lanelet_id = params.simulation.obstacle_id_to_lanelet_id[obstacle.obstacle_id][0]
                        rbg_tuple: tuple[float, float, float] = rng_cache(
                            key=obstacle_lanelet_id,
                            n=3
                        )
                        color = Color(rbg_tuple)
                        fill_color = self.obstacle_fill_color

            viewer.draw_2d_shape(
                creator=self.__class__.__name__,
                vertices=vertices,
                fill_color=fill_color if self.filled else None,
                border_color=color,
                translation=state.position,
                rotation=state.orientation,
                line_width=self.obstacle_line_width
            )

    def _render_from_graph(
        self,
        viewer: BaseViewer,
        params: RenderParams,
    ) -> None:
        if params.data is None:
            logger.warning(f"Graph data not included in rendering params - cannot render obstacles from graph.")
            return

        for index in range(params.data.v['num_nodes']):
            obstacle_id = params.data.v.id[index].item()
            if self.skip_ego_id and obstacle_id < 0:
                continue
            if obstacle_id in self.ignore_obstacle_ids:
                continue

            color = self.obstacle_color
            if self.randomize_color_from is not None:
                rng_cache = self.get_active_rng_cache(viewer, params)
                match self.randomize_color_from:
                    case "obstacle":
                        rbg_tuple: tuple[float, float, float] = rng_cache(
                            key=obstacle_id,
                            n=3
                        )
                        color = Color(rbg_tuple)
                    case "lanelet" | "viewer" | _:
                        obstacle_lanelet_assignments = torch.where(params.data.v2l.edge_index[0, :] == index)[0]
                        if len(obstacle_lanelet_assignments) > 0:
                            obstacle_lanelet_edge_idx = obstacle_lanelet_assignments[0].item()
                            obstacle_lanelet_idx = params.data.v2l.edge_index[1, obstacle_lanelet_edge_idx].item()
                            obstacle_lanelet_id = params.data.l.id[obstacle_lanelet_idx].item()
                            rbg_tuple: tuple[float, float, float] = rng_cache(
                                key=obstacle_lanelet_id,
                                n=3
                            )
                            color = Color(rbg_tuple)

            # if params.render_kwargs is not None and 'labels' in params.render_kwargs:
            #     label = str(f"{params.render_kwargs['labels'][index]:.2f}")
            # else:
            #     label = str(params.data.v.indices[index].numpy())

            if self.graph_vertices_attr is not None:
                vertices = params.data.v[self.graph_vertices_attr][index].reshape(-1, 2).numpy(force=True)
                if self.vehicle_expansion != 1.0:
                    vertices = self._expand_vertices(vertices)
                position = params.data.v[self.graph_position_attr][index].numpy(force=True)
                orientation = params.data.v[self.graph_orientation_attr][index].numpy(force=True)
                viewer.draw_2d_shape(
                    creator=self.__class__.__name__,
                    vertices=vertices,
                    fill_color=self.obstacle_fill_color if self.filled else None,
                    border_color=color,
                    translation=position,
                    rotation=orientation,
                    line_width=self.obstacle_line_width
                )
            else:
                position = params.data.v[self.graph_position_attr][index].numpy(force=True)
                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=position,
                    radius=5,
                    fill_color=color,
                    border_color=color,
                    line_width=self.obstacle_line_width,
                )
