import numpy as np

from crgeo.common.geometry.helpers import relative_orientation
from crgeo.rendering.base_renderer_plugin import BaseRendererPlugin
from crgeo.rendering.types import RenderParams
from crgeo.rendering.color_utils import T_ColorTuple
from crgeo.rendering.viewer.viewer_2d import Viewer2D


class RenderEgoVehicleInputPlugin(BaseRendererPlugin):
    def __init__(
        self,
        ego_vehicle_arrow_length: np.float = 10.0,
        ego_vehicle_orientation_arrow_color: T_ColorTuple = (0.1, 1.0, 1.0, 1.0),
        ego_vehicle_steering_angle_arrow_color: T_ColorTuple = (1.0, 0.1, 1.0, 1.0)
    ) -> None:
        self._ego_vehicle_arrow_length = ego_vehicle_arrow_length
        self._ego_vehicle_orientation_arrow_color = ego_vehicle_orientation_arrow_color
        self._ego_vehicle_steering_angle_arrow_color = ego_vehicle_steering_angle_arrow_color

    def __call__(
        self,
        viewer: Viewer2D,
        params: RenderParams
    ) -> None:
        if params.ego_vehicle is None or params.ego_vehicle.state is None:
            return

        ego_arrow_length = params.render_kwargs.pop('ego_vehicle_arrow_length', self._ego_vehicle_arrow_length)
        ego_orientation_color = params.render_kwargs.pop('ego_vehicle_orientation_arrow_color', self._ego_vehicle_orientation_arrow_color)
        ego_steering_angle_color = params.render_kwargs.pop('ego_vehicle_steering_angle_arrow_color', self._ego_vehicle_steering_angle_arrow_color)

        if 'orientation' in params.ego_vehicle.state.attributes:
            viewer.draw_arrow(
                base=params.ego_vehicle.state.position,
                angle=params.ego_vehicle.state.orientation,
                length=ego_arrow_length,
                color=ego_orientation_color
            )

        if 'steering_angle' in params.ego_vehicle.state.attributes:
            relative_steering_angle = relative_orientation(params.ego_vehicle.state.steering_angle, params.ego_vehicle.state.orientation)
            viewer.draw_arrow(
                base=params.ego_vehicle.state.position,
                angle=relative_steering_angle,
                length=ego_arrow_length,
                color=ego_steering_angle_color
            )
