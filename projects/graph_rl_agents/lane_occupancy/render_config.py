# flake8: noqa
# type: ignore

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph
from commonroad_geometric.plotting.set_research_style import set_research_style
from commonroad_geometric.rendering.plugins.base_renderer_plugin import BaseRenderPlugin
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.plugins.implementations import *
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams, SkipRenderInterrupt
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewer2D
from projects.geometric_models.lane_occupancy.utils.renderer_plugins import RenderPathOccupancyPredictionPlugin

FOLLOW_EGO = True
CAMERA_DISTANCE = 130.0
CAMERA_DISTANCE_GRAPH = 18.0
CAMERA_ROTATION = True


class PlotMatplotlibGraphRenderPlugin(BaseRenderPlugin):
    def __init__(
        self,
    ) -> None:
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(4.0, 3.3))
        # shutil.rmtree(IMAGE_EXPORT_DIR, ignore_errors=True)
        # os.makedirs(IMAGE_EXPORT_DIR, exist_ok=True)

    def __call__(
        self,
        viewer: GLViewer2D,
        params: RenderParams
    ) -> None:
        if params.time_step < 5:
            raise SkipRenderInterrupt()
        if params.time_step % 20 != 0:
            raise SkipRenderInterrupt()
        if params.ego_vehicle_simulation.current_lanelet_center_polyline.length > 20.0:
            raise SkipRenderInterrupt()
        set_research_style(size_multiplier=0.45)

        plot_road_network_graph(
            params.ego_vehicle_simulation.simulation.lanelet_graph,
            ax=self.ax,
            xlim=viewer.xlim,
            ylim=viewer.ylim,
            has_equal_axes=False,
            ignore_edge_types={LaneletEdgeType.SUCCESSOR, LaneletEdgeType.PREDECESSOR}
        )
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.tight_layout()

        # self.figure.savefig(
        #     os.path.join(IMAGE_EXPORT_DIR, 'road_network_graph_' + str(params.scenario.scenario_id) + '_' + str(params.time_step) + '_transparent.pdf'),
        #     transparent=True
        # )
        # self.figure.savefig(
        #     os.path.join(IMAGE_EXPORT_DIR, 'road_network_graph_' + str(params.scenario.scenario_id) + '_' + str(params.time_step) + '.pdf'),
        #     transparent=False
        # )


class PlotMatplotEgoPlugin(BaseRenderPlugin):
    def __init__(self) -> None:
        super().__init__()
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(4.0, 2.4))

    def __call__(
        self,
        viewer: GLViewer2D,
        params: RenderParams
    ) -> None:
        if params.time_step < 5:
            return
        if params.time_step % 20 != 0:
            return
        if params.ego_vehicle_simulation.current_lanelet_center_polyline.length > 20.0:
            return
        return
        set_research_style(size_multiplier=0.45)

        velocity = [s.velocity for s in params.ego_vehicle.state_list]
        acceleration = [s.acceleration for s in params.ego_vehicle.state_list]
        velocity = np.array(velocity[-150:])
        acceleration = np.array(acceleration[-150:])
        time_min = -len(velocity) * 0.04

        time = np.linspace(time_min, 0, len(velocity))
        max_velocity = max(12.0, velocity.max())
        xticks = np.arange(0, time_min, -1.0)[::-1]
        yticks = np.arange(0, max_velocity + 3.0, 3.0)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)
        self.ax.get_xaxis().set_major_formatter(FormatStrFormatter(r'%.0f s'))
        self.ax.get_yaxis().set_major_formatter(FormatStrFormatter(r'%d m/s'))
        self.ax.tick_params(axis='x', which='minor', direction='out', bottom=True, length=5)
        self.ax.set_ylim(0.0, max_velocity + 1.0)
        # self.ax.set_xlabel(r"$t \; [s]$")
        # self.ax.set_ylabel(r"$v_{ego} \; [m/s]$")
        self.ax.plot(time, velocity, c=(0.1, 0.8, 0.1))
        # self.ax.plot(time, acceleration, c='blue')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.tight_layout()

        self.figure.savefig(
            os.path.join(IMAGE_EXPORT_DIR, 'ego_vel' + str(params.scenario.scenario_id) +
                         '_' + str(params.time_step) + '_transparent.pdf'),
            transparent=True
        )

        self.figure.savefig(
            os.path.join(IMAGE_EXPORT_DIR, 'ego_vel' + str(params.scenario.scenario_id) +
                         '_' + str(params.time_step) + '.pdf'),
            transparent=False
        )

        self.ax.cla()


RENDERER_OPTIONS = [
    TrafficSceneRendererOptions(
        camera=EgoVehicleCamera(view_range=CAMERA_DISTANCE, camera_rotation_speed=0.7 if CAMERA_ROTATION else None) if FOLLOW_EGO else GlobalMapCamera(),
        plugins=[
            # PlotMatplotlibGraphRendererPlugin(),
            # PlotMatplotEgoPlugin(),
            RenderLaneletNetworkPlugin(lanelet_linewidth=0.64),
            RenderPlanningProblemSetPlugin(
                render_trajectory=False,
                render_start_waypoints=False,
                render_goal_waypoints=True,
                render_look_ahead_point=False
            ),
            RenderPathOccupancyPredictionPlugin(
                skip_frames=False,
                enable_plots=False,
                render_ego_encoding=True
                # enable_plots=True
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                render_trail=False,
                # trail_velocity_profile=True
            ),
            RenderObstaclePlugin(
                from_graph=False,
                randomize_color_from="obstacle",
                # obstacle_fill_color=Color((0.9, 0.9, 0.9, 1.0))
            ),
            # RenderLaneletGraphPlugin(),
            # RenderVehicleToLaneletEdgesPlugin(),
            # RenderEgoVehicleCloseupPlugin(),
        ],
    ),
]
