from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D


class RenderEgoVehicleCloseupPlugin(BaseRendererPlugin):
    def __init__(
        self,
        view_range_relative_to_ego_vehicle_length: float = 20.0
    ) -> None:
        self._view_range_relative_to_ego_vehicle_length = view_range_relative_to_ego_vehicle_length

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is not None and params.ego_vehicle.state is not None:
            center_x = params.ego_vehicle.state.position[0]
            center_y = params.ego_vehicle.state.position[1]
            view_range = params.ego_vehicle.shape.length * self._view_range_relative_to_ego_vehicle_length

            viewer.set_bounds(
                center_x - view_range / 2,
                center_x + view_range / 2,
                center_y - view_range / 2,
                center_y + view_range / 2
            )
