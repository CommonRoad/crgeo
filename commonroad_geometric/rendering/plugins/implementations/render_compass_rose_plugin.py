from dataclasses import dataclass
import math
import numpy as np

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderCompassRosePlugin(BaseRenderPlugin):
    compass_radius: float = 5.0
    compass_line_length: float = 2.0
    compass_line_width: float = 2.0
    compass_color: Color = Color((1.0, 0.0, 0.0, 1.0))  # Red

    def render(self, viewer: BaseViewer, params: RenderParams) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            return

        # Determine the position to draw the compass rose (above the vehicle)
        position_above_vehicle = np.array([
            params.ego_vehicle.state.position[0],
            params.ego_vehicle.state.position[1] + self.compass_radius + self.compass_line_length
        ])

        # Draw the compass circle
        viewer.draw_circle(
            creator=self.__class__.__name__,
            origin=position_above_vehicle,
            radius=self.compass_radius,
            border_color=self.compass_color,
            fill_color=None,
            line_width=self.compass_line_width
        )

        # Draw the compass lines for cardinal directions
        for angle_deg in [0, 90, 180, 270]:  # N, E, S, W
            angle_rad = math.radians(angle_deg)
            start_x = position_above_vehicle[0] + self.compass_radius * math.cos(angle_rad)
            start_y = position_above_vehicle[1] + self.compass_radius * math.sin(angle_rad)
            end_x = position_above_vehicle[0] + (self.compass_radius + self.compass_line_length) * math.cos(angle_rad)
            end_y = position_above_vehicle[1] + (self.compass_radius + self.compass_line_length) * math.sin(angle_rad)
            viewer.draw_arrow_from_to(
                creator=self.__class__.__name__,
                start=np.array([start_x, start_y]),
                end=np.array([end_x, end_y]),
                line_color=self.compass_color,
                line_width=self.compass_line_width
            )
