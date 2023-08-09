# flake8: noqa
# type: ignore

import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.plotting.set_research_style import set_research_style
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph
from commonroad_geometric.rendering.base_renderer_plugin import BaseRendererPlugin
from commonroad_geometric.rendering.plugins import *
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams, SkipRenderInterrupt
from commonroad_geometric.rendering.viewer.viewer_2d import Viewer2D
from projects.geometric_models.lane_occupancy.utils.renderer_plugins import RenderPathOccupancyPredictionPlugin


FOLLOW_EGO = True
CAMERA_DISTANCE = 130.0
CAMERA_DISTANCE_GRAPH = 18.0
CAMERA_ROTATION = True

class PlotMatplotlibGraphRendererPlugin(BaseRendererPlugin):
    def __init__(
        self,
    ) -> None:
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(4.0, 3.3))
        # shutil.rmtree(IMAGE_EXPORT_DIR, ignore_errors=True)
        # os.makedirs(IMAGE_EXPORT_DIR, exist_ok=True)

    def __call__(
        self,
        viewer: Viewer2D,
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

class PlotMatplotEgoPlugin(BaseRendererPlugin):
    def __init__(
        self,
    ) -> None:
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(4.0, 2.4))

    def __call__(
        self,
        viewer: Viewer2D,
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
        time_min = -len(velocity)*0.04
        
        time = np.linspace(time_min, 0, len(velocity))
        max_velocity = max(12.0, velocity.max())
        xticks = np.arange(0, time_min, -1.0)[::-1]
        yticks = np.arange(0, max_velocity+3.0, 3.0)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)
        self.ax.get_xaxis().set_major_formatter(FormatStrFormatter(r'%.0f s'))
        self.ax.get_yaxis().set_major_formatter(FormatStrFormatter(r'%d m/s'))
        self.ax.tick_params(axis='x',which='minor',direction='out',bottom=True,length=5)
        self.ax.set_ylim(0.0, max_velocity+1.0)
        #self.ax.set_xlabel(r"$t \; [s]$")
        #self.ax.set_ylabel(r"$v_{ego} \; [m/s]$")
        self.ax.plot(time, velocity, c=(0.1, 0.8, 0.1))
        #self.ax.plot(time, acceleration, c='blue')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.tight_layout()
        
        self.figure.savefig(
            os.path.join(IMAGE_EXPORT_DIR, 'ego_vel' + str(params.scenario.scenario_id) + '_' + str(params.time_step) + '_transparent.pdf'),
            transparent=True
        )
        
        self.figure.savefig(
            os.path.join(IMAGE_EXPORT_DIR, 'ego_vel' + str(params.scenario.scenario_id) + '_' + str(params.time_step) + '.pdf'),
            transparent=False
        )

        self.ax.cla()


RENDERER_OPTIONS = [
    # TrafficSceneRendererOptions(
    #     plugins=[
    #         PlotMatplotlibGraphRendererPlugin(),
    #     ],
    #     camera_follow=FOLLOW_EGO,
    #     view_range=CAMERA_DISTANCE_GRAPH,
    #     camera_auto_rotation=CAMERA_ROTATION,
    #     skip_redundant=False
    # ),
    TrafficSceneRendererOptions(
        plugins=[
            #PlotMatplotlibGraphRendererPlugin(),
            #PlotMatplotEgoPlugin(),
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
                #enable_plots=True
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                render_trail=False,
                #trail_velocity_profile=True
            ),
            RenderObstaclesPlugin(
                from_graph=False,
                randomize_color_from_lanelet=False,
                #obstacle_fill_color=(0.9, 0.9, 0.9, 1.0)
            ),
            #RenderLaneletGraphPlugin(),
            #RenderVehicleToLaneletEdgesPlugin(),
            #RenderEgoVehicleCloseupPlugin(),
        ],
        camera_follow=FOLLOW_EGO,
        view_range=CAMERA_DISTANCE if FOLLOW_EGO else None,
        camera_auto_rotation=CAMERA_ROTATION
    ),
    # TrafficSceneRendererOptions(
    #     plugins=[
    #         RenderLaneletNetworkPlugin(lanelet_linewidth=0.64),
    #         RenderPlanningProblemSetPlugin(
    #             render_trajectory=False,
    #             render_waypoints=False,
    #             render_look_ahead_point=False
    #         ),
    #         RenderPathOccupancyPredictionPlugin(
    #             skip_frames=False,
    #             enable_plots=False,
    #         ),
    #         RenderObstaclesPlugin(
    #             from_graph=False,
    #             randomize_color_from_lanelet=False,
    #             #obstacle_fill_color=(0.9, 0.9, 0.9, 1.0)
    #         ),
    #         RenderEgoVehiclePlugin(),
    #         #RenderLaneletGraphPlugin(),
    #         #RenderTrafficGraphPlugin(),
    #         #RenderVehicleToLaneletEdgesPlugin(),
    #         #RenderEgoVehicleCloseupPlugin(),
    #     ],
    #     camera_follow=FOLLOW_EGO,
    #     view_range=CAMERA_DISTANCE*2 if FOLLOW_EGO else None,
    #     camera_auto_rotation=CAMERA_ROTATION
    # ),
    # TrafficSceneRendererOptions(
    #     plugins=[
    #         RenderLaneletNetworkPlugin(
    #             fill_color=(0.94, 0.94, 0.94, 1.0),
    #             lanelet_color=(0.4, 0.4, 0.4, 0.0),
    #             lanelet_linewidth=0.64
    #         ),
    #         RenderPlanningProblemSetPlugin(
    #             render_trajectory=False,
    #             render_waypoints=False,
    #             render_look_ahead_point=False
    #         ),
    #         RenderPathOccupancyPredictionPlugin(
    #             skip_frames=False,
    #             enable_plots=False,
    #         ),
    #         RenderVehicleToLaneletEdgesPlugin(),
    #         RenderLaneletGraphPlugin(),
    #         #RenderTrafficGraphPlugin(),
    #         RenderEgoVehiclePlugin(filled=True),
    #         RenderObstaclesPlugin(
    #             filled=False,
    #             from_graph=False,
    #             obstacle_linewidth=1.0,
    #             randomize_color_from_lanelet=False,
    #             obstacle_fill_color=None#(0.9, 0.9, 0.9, 1.0)
    #         ),
    #         #RenderEgoVehicleCloseupPlugin(),
    #     ],
    #     camera_follow=FOLLOW_EGO,
    #     view_range=2*CAMERA_DISTANCE if FOLLOW_EGO else None,
    #     camera_auto_rotation=CAMERA_ROTATION
    # ),
    # TrafficSceneRendererOptions(
    #     plugins=[
    #         #RenderLaneletNetworkPlugin(),
    #         # RenderLaneletNetworkPlugin(
    #         #     fill_color=(0.96, 0.96, 0.96, 1.0),
    #         #     lanelet_color=(0.4, 0.4, 0.4, 0.0),
    #         #     lanelet_linewidth=0.64
    #         # ),
    #         RenderLaneletGraphPlugin(),
    #         RenderVehicleToLaneletEdgesPlugin(),
    #         #RenderTrafficGraphPlugin(),
    #     ],
    #     camera_follow=FOLLOW_EGO,
    #     view_range=2*CAMERA_DISTANCE if FOLLOW_EGO else None,
    #     camera_auto_rotation=CAMERA_ROTATION
    # )
]
