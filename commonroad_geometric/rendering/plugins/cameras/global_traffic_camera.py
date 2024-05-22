import numpy as np

from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import BaseCameraPlugin, CameraView2D
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


class GlobalTrafficCamera(BaseCameraPlugin):
    def __init__(
        self,
        min_view_range: float = 100.0,
        transition_speed: float = 0.05,
    ) -> None:
        """
        Args:
            min_view_range (float): Minimum view range to fall back to. Defaults to 100.0.
            transition_speed (float): Speed in [0.0, 1.0] at which camera transitions to include new vehicles.
                                      Defaults to 0.05.
        """
        self._min_view_range = min_view_range
        self._transition_speed = transition_speed
        super(GlobalTrafficCamera, self).__init__(fallback_camera=GlobalMapCamera())

    def set_camera(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        if any((
            params.simulation is None,
            not params.simulation.current_obstacles,
        )):
            # Fallback to global camera if there are no vehicles present
            self._fallback_camera(viewer, params)
            return

        current_obstacles = params.simulation.current_obstacles
        time_step = params.time_step
        min_x = min([o.state_at_time(time_step).position[0] for o in current_obstacles])
        max_x = max([o.state_at_time(time_step).position[0] for o in current_obstacles])
        min_y = min([o.state_at_time(time_step).position[1] for o in current_obstacles])
        max_y = max([o.state_at_time(time_step).position[1] for o in current_obstacles])

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_position = np.array([center_x, center_y])

        dx = max_x - min_x
        dy = max_y - min_y
        dmax = max(dx, dy) * 1.15
        view_range = max(self._min_view_range, dmax)

        if self._current_view is not None:
            previous_center_position = self._current_view.center_position
        else:
            previous_center_position = center_position
        alpha = self._transition_speed
        next_center_position = alpha * center_position + (1 - alpha) * previous_center_position

        self._current_view = CameraView2D(
            center_position=next_center_position,
            orientation=0.0,
            view_range=view_range,
        )
        viewer.set_view(camera_view=self._current_view)
