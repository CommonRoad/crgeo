from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Set

import numpy as np
import torch
import os

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.cameras.base_camera_plugin import BaseCameraPlugin
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_around_ego_plugin import RenderObstacleAroundEgoPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_flow_plugin import RenderObstacleFlowPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import CameraView2D, RenderParams
from commonroad_geometric.rendering.viewer.base_viewer import T_Viewer
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle

logger = logging.getLogger(__name__)


class CustomCamera(BaseCameraPlugin):
    def __init__(self, view_range: float):
        self.current_idx = 0
        self._view_range = view_range
        super().__init__(fallback_camera=GlobalMapCamera())

    def set_camera(self, viewer: T_Viewer, params: RenderParams):
        orientation = -params.data.v.orientation[self.current_idx].item() + np.pi / 2.0
        custom_camera_view = CameraView2D(
            center_position=params.data.v.pos[self.current_idx],
            orientation=orientation,
            view_range=self._view_range
        )
        viewer.set_view(camera_view=custom_camera_view)


class BinaryRasterizedDrivableAreaPostProcessor(BaseDataPostprocessor):
    """
    Renders the local coordinate frame drivable area as a 2D image for each vehicle.
    """

    def __init__(
        self,
        *,
        # render_size: int = 64,
        pixel_size: int = 256,
        view_range: float = 100,
        lanelet_fill_resolution: int = 20,
        lanelet_fill_offset: float = 0.4,
        flatten: bool = False,
        remove_ego: bool = False,
        only_render_ego: bool = False,
        only_incoming_edges: bool = True,
        include_road_coverage: bool = True,
        include_velocities: bool = True
    ):
        # self._render_size = render_size
        self._output_size = pixel_size
        self._view_range = view_range
        self._lanelet_fill_resolution = lanelet_fill_resolution
        self._lanelet_fill_offset = lanelet_fill_offset
        self._flatten = flatten
        self._remove_ego = remove_ego
        self._only_render_ego = only_render_ego
        self._only_incoming_edges = only_incoming_edges
        self._include_road_coverage = include_road_coverage
        self._include_velocities = include_velocities
        self._road_coverage_renderer: Optional[TrafficSceneRenderer] = None
        self._drivable_area_renderer: Optional[TrafficSceneRenderer] = None
        self._velocity_flow_renderer: Optional[TrafficSceneRenderer] = None

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert simulation is not None

        if self._drivable_area_renderer is None or not hasattr(self._drivable_area_renderer, '_viewer'):
            self._spawn_renderers()

        def velocity_flow_post_transform(frame):
            # Reverse the color calculations to get the components of relative_velocity
            occupancy_mask = torch.from_numpy((frame[:, :, 2] == 0)).int()
            relative_velocity_0 = occupancy_mask * (frame[:, :, 0]/255 - 0.5) * 100
            relative_velocity_1 = occupancy_mask * (frame[:, :, 1]/255 - 0.5) * 100
            value_tensor = torch.stack([relative_velocity_0, relative_velocity_1], dim=-1)
            return value_tensor

        renderers: List[Tuple[str, TrafficSceneRenderer]] = [
            ('drivable_area', self._drivable_area_renderer, lambda f: torch.from_numpy(np.min(f, axis=2) > 128).to(torch.uint8), (self._output_size, self._output_size)),
            ('road_coverage', self._road_coverage_renderer, lambda f: torch.from_numpy(np.min(f, axis=2) > 128).to(torch.uint8), (self._output_size, self._output_size)),
            ('velocity_flow', self._velocity_flow_renderer, velocity_flow_post_transform, (self._output_size, self._output_size, 2)),
        ]

        assert self._drivable_area_renderer is not None

        for data in samples:
            n_vehicles = data.vehicle.num_nodes

            for attr, renderer, post_transform, dims in renderers:
                if renderer is None:
                    continue

                # TODO not efficient for _only_render_ego
                value_tensor = torch.empty((n_vehicles, *dims), dtype=torch.float32)

                for idx in range(n_vehicles):
                    if self._only_render_ego and not data.vehicle.is_ego_mask[idx]:
                        continue

                    obstacle_id = data.vehicle.id[idx, 0].item()

                    if self._only_incoming_edges:
                        incoming_edge_indices = data.v2v.edge_index[0, :][data.v2v.edge_index[1, :] == idx]
                        incoming_obstacle_ids = set(data.v.id[incoming_edge_indices].flatten().tolist())
                        ignore_obstacle_ids = set(oid.item()
                                                  for oid in data.v.id if oid.item() not in incoming_obstacle_ids)
                        if not self._remove_ego:
                            ignore_obstacle_ids.discard(obstacle_id)
                    elif self._remove_ego:
                        ignore_obstacle_ids = {obstacle_id}
                    else:
                        ignore_obstacle_ids = False

                    renderer._camera.current_idx = idx
                    frame = renderer.render(
                        render_params=RenderParams(
                            time_step=data.time_step,
                            scenario=simulation.current_scenario,
                            simulation=simulation,
                            data=data,
                            render_kwargs={
                                'ignore_obstacle_ids': ignore_obstacle_ids,
                                'reference_vehicle_idx': idx
                            }
                        ),
                        return_frame=True
                    )

                    insert_idx = idx

                    # if self._output_size != self._render_size:
                    #     from PIL import Image
                    #     frame = np.array(Image.fromarray(frame).resize((self._output_size, self._output_size)))
                    value_tensor[insert_idx] = post_transform(frame)

                    if attr == 'velocity_flow':
                        if data.time_step % 20 == 0 and False:
                            import matplotlib.pyplot as plt

                            # Reverse the color calculations to get the components of relative_velocity
                            occupancy_mask = torch.from_numpy((frame[:, :, 2] == 0)).int()
                            relative_velocity_0 = occupancy_mask * (frame[:, :, 0]/255 - 0.5) * 100
                            relative_velocity_1 = occupancy_mask * (frame[:, :, 1]/255 - 0.5) * 100

                            # Create a relative velocity map for color visualization
                            masked_relative_velocity_rb_image = torch.clip(0.5 + torch.stack([relative_velocity_0, relative_velocity_1], dim=-1)/10, 0.0, 1.0)*occupancy_mask[:, :, None]

                            image = torch.cat([masked_relative_velocity_rb_image, occupancy_mask[:, :, None]], dim=-1)

                            # Prepare the grid for vector plotting
                            x, y = np.meshgrid(np.arange(relative_velocity_0.shape[1]), np.arange(relative_velocity_0.shape[0]))
                            v = relative_velocity_0.numpy()
                            u = relative_velocity_1.numpy()

                            # Plotting both the colormap and the vector field
                            fig, axes = plt.subplots(figsize=(12, 6), ncols=2)

                            # Color map visualization
                            axes[0].imshow(image.numpy())
                            axes[0].set_title('Color Map of Relative Velocity')
                            axes[0].axis('off')  # Turn off axis for cleaner visualization

                            # Vector field plot
                            quiver = axes[1].quiver(x, y, u, v, angles='xy', scale_units='xy', scale=3)
                            axes[1].invert_yaxis()
                            axes[1].set_aspect('equal')
                            axes[1].set_title('Vector Field of Relative Velocity')
                            axes[1].axis('off')  # Turn off axis for cleaner visualization

                            plt.show()

                if self._flatten:
                    data.vehicle[attr] = value_tensor.view(n_vehicles, -1)
                else:
                    data.vehicle[attr] = value_tensor

        return samples

    def _spawn_renderers(self) -> None:
        if self._include_road_coverage:
            self._road_coverage_renderer = TrafficSceneRenderer(
                options=TrafficSceneRendererOptions(
                    viewer_options=GLViewerOptions(
                        window_height=self._output_size,
                        window_width=self._output_size,
                        is_interactive=False,
                        enable_smoothing=True,
                        is_pretty=False,
                    ),
                    skip_redundant_renders=False,
                    camera=CustomCamera(view_range=self._view_range),
                    plugins=[
                        RenderLaneletNetworkPlugin(
                            fill_color=Color("black"),
                            lanelet_color=Color("black"),
                            lanelet_linewidth=0.0,
                            fill_resolution=self._lanelet_fill_resolution,
                            fill_offset=self._lanelet_fill_offset
                        ),
                    ]
                )
            )
            self._road_coverage_renderer.start()

        drivable_area_renderer_plugins = []
        if self._include_road_coverage:
            drivable_area_renderer_plugins.append(RenderLaneletNetworkPlugin(
                fill_color=Color("black"),
                lanelet_color=Color("black"),
                lanelet_linewidth=0.0,
                fill_resolution=self._lanelet_fill_resolution,
                fill_offset=self._lanelet_fill_offset
            ))
        
        drivable_area_renderer_plugins.append(RenderObstacleAroundEgoPlugin(
            obstacle_color=Color("black" if not self._include_road_coverage else "white"),
        ))

        self._drivable_area_renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                viewer_options=GLViewerOptions(
                    window_height=self._output_size,
                    window_width=self._output_size,
                    is_interactive=False,
                    enable_smoothing=False,
                    is_pretty=True,
                ),
                skip_redundant_renders=False,
                camera=CustomCamera(view_range=self._view_range),
                plugins=drivable_area_renderer_plugins
            )
        )
        self._drivable_area_renderer.start()

        if self._include_velocities:
            self._velocity_flow_renderer = TrafficSceneRenderer(
                options=TrafficSceneRendererOptions(
                    viewer_options=GLViewerOptions(
                        window_height=self._output_size,
                        window_width=self._output_size,
                        is_interactive=False,
                        enable_smoothing=False,
                        is_pretty=True,
                    ),
                    skip_redundant_renders=False,
                    camera=CustomCamera(view_range=self._view_range),
                    plugins=[
                        RenderObstacleFlowPlugin(
                            obstacle_line_width=0.0
                        ),
                    ]
                )
            )
            self._velocity_flow_renderer.start()





class RasterizedDrivableAreaPostProcessor(BaseDataPostprocessor):
    """
    Renders the local coordinate frame drivable area as a 2D image for each vehicle.
    """

    def __init__(
        self,
        *,
        frame_size: int = 256,
        view_range: float = 100,
        lanelet_fill_resolution: int = 20,
        lanelet_fill_offset: float = 0.4,
        only_ego_vehicle: bool = True,
        collect_data: bool = True,
        traffic_scene_renderer_export_dir: str = "",
        data_collection_path: str = "",
    ):
        raise NotImplementedError("WIP")
        self._frame_size = frame_size
        self._view_range = view_range
        self._lanelet_fill_resolution = lanelet_fill_resolution
        self._lanelet_fill_offset = lanelet_fill_offset
        self._only_ego_vehicle = only_ego_vehicle
        self._drivable_area_renderer: Optional[TrafficSceneRenderer] = None
        self._collect_data = collect_data
        self._traffic_scene_renderer_export_dir = traffic_scene_renderer_export_dir
        self._data_collection_path = data_collection_path
        # NOTE: The order of the items defines the class labels.
        self._class_colors: Dict[str, Color] = {
            "semi_drivable": (0, 0, 0),
            "drivable": (0.25, 0.25, 0.25),
            "vehicle": (0.5, 0.5, 0.5),
            "extrapolation": (0.75, 0.75, 0.75),
            "offroad": (1, 1, 1),
        }

        # NOTE: The frame is expected to contain the colors in range 0-255 but we
        # unfortunately cannot pass them as such.
        self._frame_colors: Dict[str, Color] = {
            "semi_drivable": (0, 0, 0),
            "drivable": (64, 64, 64),
            "vehicle": (128, 128, 128),
            "extrapolation": (191, 191, 191),
            "offroad": (255, 255, 255),
        }
        self._distinct_colors = set(self._frame_colors.values())

        self._beautiful_colors = {
            "semi_drivable": (0.0, 0.0, 0.0),
            "drivable": (0.35, 0.7, 1.0),
            "vehicle": (0.48, 0.49, 0.48),
            "extrapolation": (1.0, 1.0, 0.0),
            "offroad": (1.0, 1.0, 1.0),
        }

        self._class_colors = self._beautiful_colors

        self._spawn_drivable_area_renderer()

    def _determine_render_kwargs(
        self, data: CommonRoadData, idx: int
    ) -> Tuple[int, Set[int]]:
        obstacle_id = data.vehicle.id[idx, 0].item()

        # NOTE: Only incoming edges are considered.
        incoming_edge_indices = data.v2v.edge_index[0, :][
            data.v2v.edge_index[1, :] == idx
        ]
        incoming_obstacle_ids = set(data.v.id[incoming_edge_indices].flatten().tolist())
        ignore_obstacle_ids = set(
            oid.item() for oid in data.v.id if oid.item() not in incoming_obstacle_ids
        )

        if obstacle_id in ignore_obstacle_ids:
            ignore_obstacle_ids.remove(obstacle_id)

        return obstacle_id, ignore_obstacle_ids

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None,
    ) -> List[CommonRoadData]:
        for data in samples:
            n_vehicles = data.vehicle.num_nodes

            idx_list = range(n_vehicles)
            if self._only_ego_vehicle:
                assert ego_vehicle is not None
                assert ego_vehicle.obstacle_id is not None
                idx_list = [
                    torch.where(data.vehicle.id == ego_vehicle.obstacle_id)[0][0].item()
                ]

            value_tensor = torch.empty(
                (len(idx_list), self._frame_size, self._frame_size),
                dtype=torch.float32,
            )
            for value_tensor_idx, idx in enumerate(idx_list):
                obstacle_id, ignore_obstacle_ids = self._determine_render_kwargs(
                    data, idx
                )

                frame = self._drivable_area_renderer.render(
                    return_rgb_array=True,
                    render_params=RenderParams(
                        time_step=data.time_step,
                        scenario=simulation.current_scenario,
                        simulation=simulation,
                        data=data,
                        camera_view=CameraView(
                            position=data.v.pos[idx],
                            orientation=data.v.orientation[idx],
                            range=self._view_range,
                        ),
                        render_kwargs={
                            "ignore_obstacle_ids": ignore_obstacle_ids,
                            "ego_obstacle_id": obstacle_id,
                        },
                    ),
                )

                # frame_old = self._drivable_area_renderer_old.render(
                #     return_rgb_array=True,
                #     render_params=RenderParams(
                #         time_step=data.time_step,
                #         scenario=simulation.current_scenario,
                #         simulation=simulation,
                #         data=data,
                #         camera_view=CameraView(
                #             position=data.v.pos[idx],
                #             orientation=data.v.orientation[idx],
                #             range=64,
                #         ),
                #         render_kwargs={
                #             "ignore_obstacle_ids": ignore_obstacle_ids,
                #             "ego_obstacle_id": obstacle_id,
                #         },
                #     ),
                # )

                # frame_old = self._drivable_area_renderer_1.render(
                #     return_rgb_array=True,
                #     render_params=RenderParams(
                #         time_step=data.time_step,
                #         scenario=simulation.current_scenario,
                #         simulation=simulation,
                #         data=data,
                #         camera_view=CameraView(
                #             position=data.v.pos[idx],
                #             orientation=data.v.orientation[idx],
                #             range=64,
                #         ),
                #         render_kwargs={
                #             "ignore_obstacle_ids": ignore_obstacle_ids,
                #             "ego_obstacle_id": obstacle_id,
                #         },
                #     ),
                # )

                # frame_old = self._drivable_area_renderer_2.render(
                #     return_rgb_array=True,
                #     render_params=RenderParams(
                #         time_step=data.time_step,
                #         scenario=simulation.current_scenario,
                #         simulation=simulation,
                #         data=data,
                #         camera_view=CameraView(
                #             position=data.v.pos[idx],
                #             orientation=data.v.orientation[idx],
                #             range=64,
                #         ),
                #         render_kwargs={
                #             "ignore_obstacle_ids": ignore_obstacle_ids,
                #             "ego_obstacle_id": obstacle_id,
                #         },
                #     ),
                # )

                # frame_old = self._drivable_area_renderer_3.render(
                #     return_rgb_array=True,
                #     render_params=RenderParams(
                #         time_step=data.time_step,
                #         scenario=simulation.current_scenario,
                #         simulation=simulation,
                #         data=data,
                #         camera_view=CameraView(
                #             position=data.v.pos[idx],
                #             orientation=data.v.orientation[idx],
                #             range=64,
                #         ),
                #         render_kwargs={
                #             "ignore_obstacle_ids": ignore_obstacle_ids,
                #             "ego_obstacle_id": obstacle_id,
                #         },
                #     ),
                # )

                # frame_old = self._drivable_area_renderer_4.render(
                #     return_rgb_array=True,
                #     render_params=RenderParams(
                #         time_step=data.time_step,
                #         scenario=simulation.current_scenario,
                #         simulation=simulation,
                #         data=data,
                #         camera_view=CameraView(
                #             position=data.v.pos[idx],
                #             orientation=data.v.orientation[idx],
                #             range=64,
                #         ),
                #         render_kwargs={
                #             "ignore_obstacle_ids": ignore_obstacle_ids,
                #             "ego_obstacle_id": obstacle_id,
                #         },
                #     ),
                # )

            #     unique_rgbs = np.unique(np.reshape(frame, (-1, 3)), axis=0)
            #     distinct_frame_colors = set(map(tuple, unique_rgbs))
            #     unexpected_colors = distinct_frame_colors - self._distinct_colors
            #     assert (
            #         not unexpected_colors
            #     ), f"There are unexpected colors in the frame: {unexpected_colors}"

            #     value_array = np.zeros(
            #         (self._frame_size, self._frame_size), dtype=np.uint8
            #     )
            #     for idx, color in enumerate(self._class_colors.values()):
            #         class_mask = np.all(frame == np.array([color]), axis=-1)
            #         value_array[class_mask] = idx

            #     value_tensor[value_tensor_idx] = torch.from_numpy(value_array)
            # data.vehicle["drivable_area"] = value_tensor

            # if self._collect_data:
            #     torch.save(
            #         data,
            #         os.path.join(
            #             self._data_collection_path,
            #             f"{simulation.current_scenario.scenario_id}-{data.time_step}.pt",
            #         ),
            #     )
        return samples

    def _spawn_drivable_area_renderer(self) -> None:
        self._drivable_area_renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=self._frame_size,
                window_width=self._frame_size,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "5"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["semi_drivable"],
                        lanelet_color=self._class_colors["semi_drivable"],
                        ego_outside_map_color=self._class_colors["semi_drivable"],
                    ),
                    RenderTrafficFlowLaneletNetworkPlugin(
                        line_width=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        neighbor_lanelet_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        RenderObstaclesStyle(
                            obstacle_fill_color=self._class_colors["vehicle"],
                            obstacle_color=self._class_colors["vehicle"],
                            obstacle_linewidth=0.0,
                            render_extrapolation=True,
                            extrapolation_length=15,
                            extrapolation_color=self._class_colors["extrapolation"],
                            skip_ego_id=False,  # False is important for online rl training.
                        )
                    ),
                ],
            )
        )
        self._drivable_area_renderer.start()

        self._drivable_area_renderer_old = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=2000,
                window_width=2000,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "0"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["drivable"],
                        lanelet_color=self._class_colors["drivable"],
                        ego_outside_map_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        obstacle_fill_color=self._class_colors["offroad"],
                        obstacle_color=self._class_colors["offroad"],
                        obstacle_linewidth=0.0,
                        render_extrapolation=False,
                        skip_ego_id=False,  # False is important for online rl training.
                    ),
                ],
            )
        )

        # larger view range
        self._drivable_area_renderer_1 = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=64,
                window_width=64,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "1"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["drivable"],
                        lanelet_color=self._class_colors["drivable"],
                        ego_outside_map_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        obstacle_fill_color=self._class_colors["offroad"],
                        obstacle_color=self._class_colors["offroad"],
                        obstacle_linewidth=0.0,
                        render_extrapolation=False,
                        skip_ego_id=False,  # False is important for online rl training.
                    ),
                ],
            )
        )

        # higher resolution
        self._drivable_area_renderer_2 = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=512,
                window_width=512,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "2"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["drivable"],
                        lanelet_color=self._class_colors["drivable"],
                        ego_outside_map_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        obstacle_fill_color=self._class_colors["offroad"],
                        obstacle_color=self._class_colors["offroad"],
                        obstacle_linewidth=0.0,
                        render_extrapolation=False,
                        skip_ego_id=False,  # False is important for online rl training.
                    ),
                ],
            )
        )

        # direction of traffic
        self._drivable_area_renderer_3 = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=512,
                window_width=512,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "3"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["semi_drivable"],
                        lanelet_color=self._class_colors["semi_drivable"],
                        ego_outside_map_color=self._class_colors["semi_drivable"],
                    ),
                    RenderTrafficFlowLaneletNetworkPlugin(
                        line_width=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        neighbor_lanelet_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        obstacle_fill_color=self._class_colors["offroad"],
                        obstacle_color=self._class_colors["offroad"],
                        obstacle_linewidth=0.0,
                        render_extrapolation=False,
                        skip_ego_id=False,  # False is important for online rl training.
                    ),
                ],
            )
        )

        # dynamic obstacles vs offroad
        self._drivable_area_renderer_4 = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=512,
                window_width=512,
                export_dir=os.path.join(self._traffic_scene_renderer_export_dir, "4"),
                interactive=False,
                skip_redundant=True,
                smoothing=False,
                pretty=False,
                plugins=[
                    RenderLaneletNetworkPlugin(
                        lanelet_linewidth=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        fill_color=self._class_colors["semi_drivable"],
                        lanelet_color=self._class_colors["semi_drivable"],
                        ego_outside_map_color=self._class_colors["semi_drivable"],
                    ),
                    RenderTrafficFlowLaneletNetworkPlugin(
                        line_width=0.0,
                        fill_resolution=self._lanelet_fill_resolution,
                        fill_offset=self._lanelet_fill_offset,
                        neighbor_lanelet_color=self._class_colors["drivable"],
                    ),
                    RenderObstaclePlugin(
                        obstacle_fill_color=self._class_colors["vehicle"],
                        obstacle_color=self._class_colors["vehicle"],
                        obstacle_linewidth=0.0,
                        render_extrapolation=False,
                        skip_ego_id=False,  # False is important for online rl training.
                    ),
                ],
            )
        )
