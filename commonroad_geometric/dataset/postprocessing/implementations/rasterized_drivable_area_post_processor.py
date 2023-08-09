from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import torch
import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin, RenderObstaclesStyle
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import CameraView, RenderParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


logger = logging.getLogger(__name__)


class RasterizedDrivableAreaPostProcessor(BaseDataPostprocessor):
    """
    Renders the local coordinate frame drivable area as a 2D image for each vehicle.
    """

    def __init__(
        self,
        *,
        #render_size: int = 64,
        pixel_size: int = 256,
        view_range: float = 100,
        lanelet_fill_resolution: int = 20,
        lanelet_fill_offset: float = 0.4,
        flatten: bool = False,
        remove_ego: bool = False,
        only_incoming_edges: bool = True
    ):
        #self._render_size = render_size
        self._output_size = pixel_size
        self._view_range = view_range
        self._lanelet_fill_resolution = lanelet_fill_resolution
        self._lanelet_fill_offset = lanelet_fill_offset
        self._flatten = flatten
        self._remove_ego = remove_ego
        self._only_incoming_edges = only_incoming_edges
        self._road_coverage_renderer: Optional[TrafficSceneRenderer] = None
        self._drivable_area_renderer: Optional[TrafficSceneRenderer] = None

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert simulation is not None

        if self._road_coverage_renderer is None:
            self._spawn_renderers()

        renderers: List[Tuple[str, TrafficSceneRenderer]] = [
            ('drivable_area', self._drivable_area_renderer), 
            ('road_coverage', self._road_coverage_renderer)
        ]

        assert self._drivable_area_renderer is not None
        assert self._road_coverage_renderer is not None

        for data in samples:
            n_vehicles = data.vehicle.num_nodes

            for attr, renderer in renderers:
                value_tensor = torch.empty((n_vehicles, self._output_size, self._output_size), dtype=torch.float32)
            
                for idx in range(n_vehicles):
                    obstacle_id = data.vehicle.id[idx, 0].item()

                    if self._only_incoming_edges:
                        incoming_edge_indices = data.v2v.edge_index[0, :][data.v2v.edge_index[1, :] == idx]
                        incoming_obstacle_ids = set(data.v.id[incoming_edge_indices].flatten().tolist())
                        ignore_obstacle_ids = set(oid.item() for oid in data.v.id if oid.item() not in incoming_obstacle_ids)
                        if not self._remove_ego:
                            ignore_obstacle_ids.remove(obstacle_id)
                    elif self._remove_ego:
                        ignore_obstacle_ids = {obstacle_id}
                    else:
                        ignore_obstacle_ids = False

                    frame = renderer.render(
                        return_rgb_array=True,
                        render_params=RenderParams(
                            time_step=data.time_step,
                            scenario=simulation.current_scenario,
                            simulation=simulation,
                            data=data,
                            camera_view=CameraView(
                                position=data.v.pos[idx],
                                orientation=data.v.orientation[idx],
                                range=self._view_range
                            ),
                            render_kwargs={
                                'ignore_obstacle_ids': ignore_obstacle_ids
                            }
                        )
                    )


                    # if self._output_size != self._render_size:
                    #     from PIL import Image
                    #     frame = np.array(Image.fromarray(frame).resize((self._output_size, self._output_size)))
                    value_tensor[idx] = torch.from_numpy(np.min(frame, axis=2) < 255).float()
                    
                    # import matplotlib.pyplot as plt
                    # fig, axes = plt.subplots(figsize=(6, 3.2), ncols=2)
                    # axes[0].set_title(attr + 'raw')
                    # axes[0].imshow(frame)
                    # axes[1].set_title(attr)
                    # axes[1].imshow(value_tensor[idx])
                    # #ax.set_aspect('equal')
                    # plt.show()

            
                if self._flatten:
                    data.vehicle[attr] = value_tensor.view(n_vehicles, -1)
                else:
                    data.vehicle[attr] = value_tensor
            
        return samples

    def _spawn_renderers(self) -> None:
        self._road_coverage_renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=self._output_size,
                window_width=self._output_size,
                interactive=False,
                skip_redundant=False,
                smoothing=True,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        fill_color=(0,0,0),
                        lanelet_color=(0,0,0),
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset
                    ),
                ]
            )
        )
        self._road_coverage_renderer.start()

        self._drivable_area_renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=self._output_size,
                window_width=self._output_size,
                interactive=False,
                skip_redundant=False,
                smoothing=False,
                pretty=True,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        fill_color=(0,0,0),
                        lanelet_color=(0,0,0),
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset
                    ),
                    RenderObstaclesPlugin(RenderObstaclesStyle(
                        obstacle_fill_color=(1,1,1),
                        obstacle_color=(1,1,1),
                        obstacle_linewidth=0.0
                    )),
                ]
            )
        )
        self._drivable_area_renderer.start()

