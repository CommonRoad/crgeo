import numpy as np

from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer


class RenderTrajectoryPredictionPlugin(BaseRenderPlugin):

    def __init__(
        self,
        render_trajectory_waypoints: bool = False,
        render_smoothed_prediction_waypoints: bool = True,
        render_prediction_waypoints: bool = False,
        render_start_node: bool = True
    ):
        self.render_trajectory_waypoints = render_trajectory_waypoints
        self.render_smoothed_prediction = render_smoothed_prediction_waypoints
        self.render_prediction_waypoints = render_prediction_waypoints
        self.render_start_node = render_start_node

    def render(self, viewer: T_Viewer, params: RenderParams) -> None:
        assert isinstance(params.data, CommonRoadData)
        trajectory_prediction_container = params.render_kwargs['output'][0]
        trajectory_prediction = trajectory_prediction_container.prediction
        # data = params.data[params.time_step]
        # vehicle_ids = trajectory_prediction_container.vehicle_ids_tensor
        # vehicle_mask = torch.any(data.vehicle.id == vehicle_ids, dim=1)
        # assert params.trajectory_prediction is not None and len(params.trajectory_prediction) > params.time_step

        start_t = trajectory_prediction_container.rel_time_slice_pred.start - 1
        start_data = params.data.vehicle_at_time_step(start_t)

        trajectory_data = trajectory_prediction_container.data_window_list[trajectory_prediction_container.rel_time_slice_pred]

        for vehicle_idx in range(trajectory_prediction_container.vehicle_ids_pred.shape[0]):
            matches = (start_data.id == trajectory_prediction_container.vehicle_ids_pred[vehicle_idx]).squeeze(-1)
            start_pos = start_data.pos[matches].cpu().numpy()
            predicted_pos = np.array([
                trajectory_prediction[t].position[vehicle_idx].detach().cpu().numpy()
                for t in range(len(trajectory_prediction))
            ])
            # viewer.draw_polyline(polyline.waypoints)

            assert np.linalg.norm(start_pos - predicted_pos[0]) < 10.0

            if self.render_smoothed_prediction:
                waypoints = np.vstack([start_pos, predicted_pos])
                polyline = ContinuousPolyline(waypoints)
                for t in range(polyline.waypoints.shape[0] - 1):
                    this_pos = polyline.waypoints[t]
                    next_pos = polyline.waypoints[t + 1]
                    t_rel = t / polyline.waypoints.shape[0]
                    color = Color((1 - t_rel, 0.0, t_rel, 0.6))
                    viewer.draw_line(
                        creator=self.__class__.__name__,
                        start=this_pos,
                        end=next_pos,
                        color=color,
                        line_width=0.75,
                    )

            for t in range(predicted_pos.shape[0]):
                t_rel = t / predicted_pos.shape[0]
                color = Color((1 - t_rel, 0.0, t_rel, 0.6))

                if self.render_prediction_waypoints:
                    viewer.draw_circle(
                        creator=self.__class__.__name__,
                        origin=predicted_pos[t],
                        radius=0.3,
                        fill_color=color,
                        border_color=Color((0.0, 0.0, 0.0)),
                    )

                if self.render_trajectory_waypoints:
                    matches = (trajectory_data[t].v.id == trajectory_prediction_container.vehicle_ids_pred[vehicle_idx]).squeeze(-1)
                    real_pos = trajectory_data[t].v.pos[matches].cpu().numpy()

                    if self.render_prediction_waypoints:
                        viewer.draw_circle(
                            creator=self.__class__.__name__,
                            origin=real_pos[0],
                            radius=0.3,
                            fill_color=None,
                            border_color=color,
                            line_width=0.3
                        )
                    else:
                        viewer.draw_circle(
                            creator=self.__class__.__name__,
                            origin=real_pos[0],
                            radius=0.3,
                            fill_color=color,
                            border_color=Color((0.0, 0.0, 0.0)),
                        )

            if self.render_start_node:
                viewer.draw_circle(
                    creator=self.__class__.__name__,
                    origin=start_pos[0],
                    radius=0.4,
                    fill_color=Color((1.0, 0.0, 0.0), alpha=1.0),
                    border_color=Color((0.0, 0.0, 0.0)),
                )
