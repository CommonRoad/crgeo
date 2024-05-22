from dataclasses import dataclass

from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import BaseViewer


@dataclass
class RenderEgoVehicleInputPlugin(BaseRenderPlugin):
    ego_vehicle_arrow_length: float = 10.0
    ego_vehicle_orientation_arrow_color: Color = Color((0.1, 1.0, 1.0, 1.0))
    ego_vehicle_steering_angle_arrow_color: Color = Color((1.0, 0.1, 1.0, 1.0))

    def render(
        self,
        viewer: BaseViewer,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            return

        if 'orientation' in params.ego_vehicle.state.attributes:
            viewer.draw_2d_arrow(
                creator=self.__class__.__name__,
                origin=params.ego_vehicle.state.position,
                angle=params.ego_vehicle.state.orientation,
                length=self.ego_vehicle_arrow_length,
                line_color=self.ego_vehicle_orientation_arrow_color
            )

        if 'steering_angle' in params.ego_vehicle.state.attributes:
            relative_steering_angle = relative_orientation(params.ego_vehicle.state.steering_angle,
                                                           params.ego_vehicle.state.orientation)
            viewer.draw_2d_arrow(
                creator=self.__class__.__name__,
                origin=params.ego_vehicle.state.position,
                angle=relative_steering_angle,
                length=self.ego_vehicle_arrow_length,
                line_color=self.ego_vehicle_steering_angle_arrow_color
            )
