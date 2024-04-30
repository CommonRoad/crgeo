from typing import Optional

import numpy as np

from commonroad_geometric.common.geometry.helpers import princip
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import BaseCameraPlugin, CameraView2D
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


class EgoVehicleCamera(BaseCameraPlugin):
    def __init__(
        self,
        view_range: float = 150.0,
        camera_rotation_speed: Optional[float] = 2.5,
    ) -> None:
        self._view_range = view_range
        self._camera_rotation_speed = camera_rotation_speed
        self._rotation = 0
        super(EgoVehicleCamera, self).__init__(fallback_camera=GlobalMapCamera())

    def set_camera(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            # Fallback to global camera if there are no ego vehicles present
            self._fallback_camera(viewer, params)
            return

        if self._camera_rotation_speed is None:
            self._rotation = -params.ego_vehicle.state.orientation + np.pi / 2.0
        else:
            self._rotation += params.simulation.dt * self._camera_rotation_speed * princip(
                -params.ego_vehicle.state.orientation + np.pi / 2.0 - self._rotation
            )

        self._current_view = CameraView2D(
            center_position=params.ego_vehicle.state.position,
            orientation=self._rotation,
            view_range=self._view_range
        )
        viewer.set_view(
            camera_view=self._current_view
        )
