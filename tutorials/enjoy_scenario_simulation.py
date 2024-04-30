import sys, os; sys.path.insert(0, os.getcwd())

import argparse
import shutil
from functools import partial
from pathlib import Path

from commonroad_geometric.common.logging import stdout
from commonroad_geometric.common.utils.seeding import set_global_seed
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations import OverlappingTrajectoriesFilter
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import DepopulateScenarioPreprocessor, LaneletNetworkSubsetPreprocessor, VehicleFilterPreprocessor
from commonroad_geometric.rendering import Color
from commonroad_geometric.rendering.plugins.cameras.follow_vehicle_camera import FollowVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletGraphPlugin, RenderTrafficGraphPlugin, RenderVehicleToLaneletEdgesPlugin
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_trails_plugin import RenderObstacleTrailPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions

# INPUT_SCENARIO = Path('data/other/')
# INPUT_SCENARIO = Path('data/t_junction_test')
# INPUT_SCENARIO = Path('data/t_junction_recorded')
# INPUT_SCENARIO = Path('data/other_recordings')
# INPUT_SCENARIO = Path('data/highway_test')
INPUT_SCENARIO = Path('data/osm_recordings')

EXPORT_DIR = Path('outputs/enjoy')
EDGE_DRAWER = VoronoiEdgeDrawer
EDGE_DISTANCE_THRESHOLD = 30.0
EXTRACT_TEMPORAL = False
TEMPORAL_TIME_STEPS = 10
TEMPORAL_ALPHA_MULTIPLIER = 0.7
RENDERER_SIZE = (1000, 800)
SCREENSHOT_RATE = 20
HD_RESOLUTION_MULTIPLIER = 2.0
VIEW_RANGE = 75.0
FPS = 60
TRANSPARENT_SCREENSHOTS = True


def create_renderers(args) -> dict[str, TrafficSceneRenderer]:
    renderers: dict[str, TrafficSceneRenderer] = {}
    renderers['traffic'] = TrafficSceneRenderer(
        options=TrafficSceneRendererOptions(
            viewer_options=GLViewerOptions(
                caption="CommonRoad Viewer",
                window_width=RENDERER_SIZE[0],
                window_height=RENDERER_SIZE[1],
                window_scaling_factor=HD_RESOLUTION_MULTIPLIER if args.hd else 1.0,
                # transparent_screenshots=TRANSPARENT_SCREENSHOTS,
            ),
            camera=FollowVehicleCamera(view_range=VIEW_RANGE) if args.camera_follow else GlobalMapCamera(),
            plugins=[
                RenderLaneletNetworkPlugin(randomize_lanelet_color=True),
                RenderObstaclePlugin(
                    skip_ego_id=False,
                    randomize_color_from="viewer"
                ),
                RenderObstacleTrailPlugin(
                    randomize_color_from="viewer",
                    trail_interval=10,
                    trail_alpha=True
                ),
                RenderTrafficGraphPlugin()
            ],
        )
    )

    if args.extract:
        # renderers['geometric'] = TrafficSceneRenderer(
        #     options=TrafficSceneRendererOptions(
        #         viewer_options=GLViewerOptions(
        #             caption="Geometric Viewer",
        #             window_width=RENDERER_SIZE[0],
        #             window_height=RENDERER_SIZE[1],
        #             window_scaling_factor=HD_RESOLUTION_MULTIPLIER if args.hd else 1.0,
        #             transparent_screenshots=TRANSPARENT_SCREENSHOTS,
        #         ),
        #         camera=FollowVehicleCamera(view_range=VIEW_RANGE) if args.camera_follow else GlobalMapCamera(),
        #         plugins=[
        #             RenderLaneletGraphPlugin(),
        #             RenderVehicleToLaneletEdgesPlugin(
        #                 edge_arc=0.05,
        #                 temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
        #             ),
        #             RenderTrafficGraphPlugin(
        #                 edge_color_other_connection=Color((0.0, 0.9, 0.0, 0.25)),
        #                 edge_arc=0.0,
        #                 node_radius=0.65,
        #                 node_fillcolor=Color((0.0, 0.9, 0.0, 0.6)),
        #                 temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
        #             ),
        #         ],
        #     )
        # )
        renderers['all'] = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                viewer_options=GLViewerOptions(
                    caption="Geometric Viewer",
                    window_width=RENDERER_SIZE[0],
                    window_height=RENDERER_SIZE[1],
                    window_scaling_factor=HD_RESOLUTION_MULTIPLIER if args.hd else 1.0,
                    transparent_screenshots=TRANSPARENT_SCREENSHOTS,
                ),
                camera=FollowVehicleCamera(view_range=VIEW_RANGE) if args.camera_follow else GlobalMapCamera(),
                plugins=[
                    RenderLaneletNetworkPlugin(from_graph=True),
                    RenderLaneletGraphPlugin(),
                    RenderObstaclePlugin(
                        skip_ego_id=False,
                        from_graph=True
                    ),
                    RenderVehicleToLaneletEdgesPlugin(
                        edge_arc=0.05,
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                    RenderTrafficGraphPlugin(
                        edge_color_other_connection=Color((0.0, 0.9, 0.0, 0.25)),
                        edge_arc=0.0,
                        node_radius=0.65,
                        node_fillcolor=Color((0.0, 0.9, 0.0, 0.6)),
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                ],
            )
        )
        # renderers['traffic_graph'] = TrafficSceneRenderer(
        #     options=TrafficSceneRendererOptions(
        #         viewer_options=GLViewerOptions(
        #             caption="Traffic Graph Viewer",
        #             window_width=RENDERER_SIZE[0],
        #             window_height=RENDERER_SIZE[1],
        #             window_scaling_factor=HD_RESOLUTION_MULTIPLIER if args.hd else 1.0,
        #             transparent_screenshots=TRANSPARENT_SCREENSHOTS,
        #         ),
        #         camera=FollowVehicleCamera(view_range=VIEW_RANGE) if args.camera_follow else GlobalMapCamera(),
        #         plugins=[
        #             RenderTrafficGraphPlugin(
        #                 edge_color_other_connection=Color((0.0, 0.9, 0.0, 0.25)),
        #                 edge_arc=0.0,
        #                 node_radius=0.65,
        #                 node_fillcolor=Color((0.0, 0.9, 0.0, 0.6)),
        #                 temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
        #             ),
        #         ],
        #     )
        # )
        # renderers['lanelet_graph'] = TrafficSceneRenderer(
        #     options=TrafficSceneRendererOptions(
        #         viewer_options=GLViewerOptions(
        #             caption="Lanelet Graph Viewer",
        #             window_width=RENDERER_SIZE[0],
        #             window_height=RENDERER_SIZE[1],
        #             window_scaling_factor=HD_RESOLUTION_MULTIPLIER if args.hd else 1.0,
        #             transparent_screenshots=TRANSPARENT_SCREENSHOTS,
        #         ),
        #         camera=FollowVehicleCamera(view_range=VIEW_RANGE) if args.camera_follow else GlobalMapCamera(),
        #         plugins=[
        #             RenderLaneletGraphPlugin(
        #                 edge_arc=0.1
        #             ),
        #         ],
        #     )
        # )

    return renderers


def enjoy(args) -> None:
    if args.screenshots:
        shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    renderers = create_renderers(args)

    preprocessing_pipeline = VehicleFilterPreprocessor()
    preprocessing_pipeline >>= LaneletNetworkSubsetPreprocessor(radius=500.0)
    preprocessing_pipeline >>= DepopulateScenarioPreprocessor(5)
    preprocessing_pipeline >>= OverlappingTrajectoriesFilter()

    scenario_iterator = ScenarioIterator(
        directory=args.scenario_dir,
        preprocessor=preprocessing_pipeline,
    )

    simulation_cls = SumoSimulation if args.interactive else ScenarioSimulation
    simulation_options_cls = partial(
        SumoSimulationOptions,
        presimulation_steps=0
    ) if args.interactive else ScenarioSimulationOptions

    print(f"Enjoying {args.scenario_dir} ({scenario_iterator.max_result_scenarios} scenarios)")
    print(f"Preprocessing strategy: {preprocessing_pipeline.name}")

    frame_count: int = 0
    for scenario_bundle in scenario_iterator:
        simulation = simulation_cls(
            initial_scenario=scenario_bundle.preprocessed_scenario,
            options=simulation_options_cls(
                backup_current_scenario=False,
                backup_initial_scenario=False,
                dt=args.dt
            )
        )

        print(f"Enjoying {scenario_bundle.scenario_path}")

        if args.extract:
            if EXTRACT_TEMPORAL:
                sub_extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=EDGE_DRAWER(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
                extractor = TemporalTrafficExtractor(
                    traffic_extractor=sub_extractor,
                    options=TemporalTrafficExtractorOptions(
                        collect_num_time_steps=TEMPORAL_TIME_STEPS,
                        collect_skip_time_steps=0,
                        return_incomplete_temporal_graph=True
                    )
                )
            else:
                extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=EDGE_DRAWER(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
        else:
            extractor = None

        simulation.start()
        for time_step, scenario in simulation:
            if args.extract:
                data = extractor.extract(
                    time_step=time_step
                )
                # print(data)
            else:
                data = None

            # obstacle = simulation.current_obstacles[0]
            # renderer.set_view(
            #     obstacle.state_at_time(time_step).position,
            #     obstacle.state_at_time(time_step).orientation,
            #     range=100.0
            # )

            capture_screenshot = args.screenshots and frame_count % SCREENSHOT_RATE == 0
            if capture_screenshot:
                for renderer_name, renderer in renderers.items():
                    renderer.screenshot(
                        output_file=EXPORT_DIR.joinpath(f"enjoy_{frame_count}_{renderer_name}.png"),
                    )
            # Contains one frame per renderer
            frames = simulation.render(
                renderers=list(renderers.values()),
                return_frames=True,
                render_params=RenderParams(
                    time_step=time_step,
                    scenario=scenario,
                    data=data
                ),
                **{"obstacle_color": Color("white")}
            )

            stdout(f"Enjoying {scenario.scenario_id} (timestep {time_step}/{simulation.final_time_step})")
            if args.max_timesteps is not None and time_step >= args.max_timesteps:
                break

            frame_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play scenario videos.")
    parser.add_argument("--scenario-dir", type=str, default=INPUT_SCENARIO, help="path to scenario directory or scenario file")
    parser.add_argument("--max-timesteps", type=int, help="optional maximum number of timesteps per scenario")
    parser.add_argument("--extract", action="store_true", help="whether to extract traffic graph data")
    parser.add_argument("--camera-follow", action="store_true", help="whether to let the camera follow a vehicle")
    parser.add_argument("--interactive", action="store_true", help="whether to simulate traffic using SUMO")
    parser.add_argument("--dt", type=float, help="time delta for simulation steps", default=0.04)
    parser.add_argument("--screenshots", action="store_true", help="capture and export rendering screenshots")
    parser.add_argument("--hd", action="store_true", help="high resolution rendering")

    args = parser.parse_args()
    set_global_seed(0)
    enjoy(args)
