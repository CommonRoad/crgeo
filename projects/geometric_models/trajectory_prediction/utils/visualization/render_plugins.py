from typing import Tuple

from math import sqrt
import numpy as np
import torch
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from pyglet import gl
from pyglet.image import ImageData
from torch import Tensor




class RenderTrajectoryPredictionPlugin(BaseRendererPlugin):

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

    def __call__(self, viewer: Viewer2D, params: RenderParams) -> None:
        assert isinstance(params.data, CommonRoadData)
        trajectory_prediction_container = params.render_kwargs['output'][0]
        trajectory_prediction = trajectory_prediction_container.prediction
        #data = params.data[params.time_step]
        #vehicle_ids = trajectory_prediction_container.vehicle_ids_tensor
        # vehicle_mask = torch.any(data.vehicle.id == vehicle_ids, dim=1)
        # assert params.trajectory_prediction is not None and len(params.trajectory_prediction) > params.time_step

        start_t = trajectory_prediction_container.rel_time_slice_pred.start - 1
        start_data = params.data.vehicle_at_time_step(start_t)

        trajectory_data = trajectory_prediction_container.data_window_list[trajectory_prediction_container.rel_time_slice_pred]

        for vehicle_idx in range(trajectory_prediction_container.vehicle_ids_pred.shape[0]):
            start_pos = start_data.pos[(start_data.id == trajectory_prediction_container.vehicle_ids_pred[vehicle_idx]).squeeze(-1)].cpu().numpy()
            predicted_pos = np.array([trajectory_prediction[t].position[vehicle_idx].detach().cpu().numpy() for t in range(len(trajectory_prediction))])
            #viewer.draw_polyline(polyline.waypoints)

            assert np.linalg.norm(start_pos - predicted_pos[0]) < 10.0

            if self.render_smoothed_prediction:
                waypoints = np.vstack([start_pos, predicted_pos])
                polyline = ContinuousPolyline(waypoints)
                for t in range(polyline.waypoints.shape[0] - 1):
                    this_pos = polyline.waypoints[t]
                    next_pos = polyline.waypoints[t + 1]
                    t_rel = t/polyline.waypoints.shape[0]
                    color = (1 - t_rel, 0.0, t_rel, 0.6)
                    viewer.draw_line(
                        this_pos,
                        next_pos,
                        linewidth=0.75,
                        color=color
                    )

            for t in range(predicted_pos.shape[0]):
                t_rel = t/predicted_pos.shape[0]
                color = (1 - t_rel, 0.0, t_rel, 0.6)
                
                if self.render_prediction_waypoints:
                    viewer.draw_circle(
                        origin=predicted_pos[t],
                        radius=0.3,
                        color=color,
                        outline=False,
                        linecolor=(0.0,0.0,0.0),
                        linewidth=None
                    )

                if self.render_trajectory_waypoints:
                    real_pos = trajectory_data[t].v.pos[(trajectory_data[t].v.id == trajectory_prediction_container.vehicle_ids_pred[vehicle_idx]).squeeze(-1)].cpu().numpy()
                    
                    if self.render_prediction_waypoints:
                        viewer.draw_circle(
                            origin=real_pos[0],
                            radius=0.3,
                            filled=False,
                            color=color,
                            linewidth=0.3
                        )
                    else:
                        viewer.draw_circle(
                            origin=real_pos[0],
                            radius=0.3,
                            color=color,
                            outline=False,
                            linecolor=(0.0,0.0,0.0),
                            linewidth=None
                        )
            
            if self.render_start_node:
                viewer.draw_circle(
                    origin=start_pos[0],
                    radius=0.4,
                    color=(1.0, 0.0, 0.0, 1.0),
                    outline=False,
                    linecolor=(0.0,0.0,0.0),
                    linewidth=None
                )