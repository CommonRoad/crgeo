from typing import Optional

import numpy as np
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.geometry import ContinuousPolyline
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import BaseCameraPlugin, CameraView2D
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer
from commonroad_geometric.rendering.viewer.utils import transform_vertices_2d


class GlobalMapCamera(BaseCameraPlugin):
    def __init__(
        self,
        view_range: Optional[float] = None
    ):
        self._view_range = view_range
        super(GlobalMapCamera, self).__init__(fallback_camera=self)

    def set_camera(
        self,
        viewer: T_Viewer,
        params: RenderParams
    ) -> None:
        lanelet_network = params.scenario.lanelet_network
        min_x = min([ll.center_vertices[:, 0].min() for ll in lanelet_network.lanelets])
        max_x = max([ll.center_vertices[:, 0].max() for ll in lanelet_network.lanelets])
        min_y = min([ll.center_vertices[:, 1].min() for ll in lanelet_network.lanelets])
        max_y = max([ll.center_vertices[:, 1].max() for ll in lanelet_network.lanelets])

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_position = np.array([center_x, center_y])

        lanelet_paths = {ll.lanelet_id: ContinuousPolyline(ll.center_vertices) for ll in lanelet_network.lanelets}
        distances = {
            lid: path.get_projected_distance(center_position, linear_projection=True)
            for lid, path in lanelet_paths.items()
        }
        closest_lanelet_id = min(distances.keys(), key=lambda x: distances[x])
        closest_lanelet_orientation = lanelet_paths[closest_lanelet_id].get_projected_direction(
            center_position,
            linear_projection=True
        )
        rotation = -closest_lanelet_orientation + np.pi / 2
        vertices = np.array([[min_x, min_y], [max_x, max_y]])
        rotated_vertices = transform_vertices_2d(
            vertices=vertices,
            rotation=rotation
        )
        (min_x, min_y), (max_x, max_y) = rotated_vertices

        dx = abs(max_x - min_x)
        dy = abs(max_y - min_y)
        dmax = max(dx, dy) * 1.15
        view_range = dmax if self._view_range is None else self._view_range

        self._current_view = CameraView2D(
            center_position=center_position,
            orientation=rotation,
            view_range=view_range,
        )
        viewer.set_view(camera_view=self._current_view)
