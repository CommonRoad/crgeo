from typing import Optional

import numpy as np

from commonroad_geometric.common.geometry.helpers import princip
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import BaseCameraPlugin, CameraView2D
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


class FollowVehicleCamera(BaseCameraPlugin):
    def __init__(
        self,
        view_range: float = 150.0,
        camera_rotation_speed: Optional[float] = 0.7,
    ):
        self._view_range = view_range
        self._camera_rotation_speed = camera_rotation_speed
        self._rotation: Optional[float] = None
        super(FollowVehicleCamera, self).__init__(fallback_camera=GlobalMapCamera())

    def set_camera(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        if params.simulation is None or not params.simulation.current_obstacles:
            # Fallback to global camera if there are no vehicles present
            self._fallback_camera(viewer, params)
            return

        follow_state = params.simulation.current_obstacles[0].state_at_time(
            params.time_step
        )

        if self._rotation is None:
            self._rotation = follow_state.orientation

        if self._camera_rotation_speed is None:
            self._rotation = follow_state.orientation
        else:
            self._rotation += params.simulation.dt * self._camera_rotation_speed * princip(
                -follow_state.orientation + np.pi / 2.0 - self._rotation
            )

        self._current_view = CameraView2D(
            center_position=follow_state.position,
            orientation=self._rotation,
            view_range=self._view_range
        )
        viewer.set_view(
            camera_view=self._current_view
        )
