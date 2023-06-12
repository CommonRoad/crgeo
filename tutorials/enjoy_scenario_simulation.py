from functools import partial
import sys, os
from typing import List; sys.path.insert(0, os.getcwd())

import argparse
import shutil

from commonroad_geometric.common.utils.seeding import set_global_seed
from commonroad_geometric.common.logging import stdout
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.iteration import ScenarioIterator
from commonroad_geometric.dataset.preprocessing.implementations import *
from commonroad_geometric.rendering.plugins import *
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions

INPUT_SCENARIO = 'data/osm_recordings'

PREPROCESSORS = [
    #LaneletNetworkSubsetPreprocessor(radius=95.0),
    # RemoveIslandsPreprocessor(),
    #SegmentLaneletsPreprocessor(lanelet_max_segment_length=25.0),
    #VehicleFilterPreprocessor(),
    #CutLeafLaneletsPreprocessor(cutoff=50.0)
    #DepopulateScenarioPreprocessor(5)  
]

FILTERS = [
    # ValidTrajectoriesFilterer()
]

EXPORT_DIR = 'outputs/enjoy'
EDGE_DRAWER = NoEdgeDrawer
EDGE_DISTANCE_THRESHOLD = 30.0
EXTRACT_TEMPORAL = False
TEMPORAL_TIME_STEPS = 10
TEMPORAL_ALPHA_MULTIPLIER = 0.7
RENDERER_SIZE = (1000, 800)
SCREENSHOT_RATE = 20
HD_RESOLUTION_MULTIPLIER = 5.0
VIEW_RANGE = 75.0
FPS = 60
TRANSPARENT_SCREENSHOTS = True


def create_renderers(args) -> dict[str, TrafficSceneRenderer]:
    options = dict(
        window_width=RENDERER_SIZE[0]*HD_RESOLUTION_MULTIPLIER if args.hd else RENDERER_SIZE[0],
        window_height=RENDERER_SIZE[1]*HD_RESOLUTION_MULTIPLIER if args.hd else RENDERER_SIZE[1],
        camera_follow=args.camera_follow,
        view_range=VIEW_RANGE, # if args.camera_follow else None,
        fps=FPS,
        transparent_screenshots=TRANSPARENT_SCREENSHOTS,
        camera_auto_rotation=True
    )

    renderers: dict[str, TrafficSceneRenderer] = {}
    renderers['traffic'] = TrafficSceneRenderer(
        options=TrafficSceneRendererOptions(
            plugins=[
                RenderLaneletNetworkPlugin(),
                RenderObstaclesPlugin(
                    skip_ego_id=False
                ),
            ],
            caption="CommonRoad Viewer",
            **options
        )
    )

    if args.extract:
        renderers['geometric'] = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                plugins=[
                    RenderLaneletGraphPlugin(),
                    RenderVehicleToLaneletEdgesPlugin(
                        edge_arc=0.05,
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                    RenderTrafficGraphPlugin(
                        edge_color_other_connection=(0.0, 0.9, 0.0, 0.25),
                        edge_arc=0.0,
                        node_radius=0.65,
                        node_fillcolor=(0.0, 0.9, 0.0, 0.6),
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                ],
                caption="Geometric Viewer",
                **options
            )
        )
        renderers['all'] = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                plugins=[
                    RenderLaneletNetworkPlugin(),
                    RenderLaneletGraphPlugin(),
                    RenderObstaclesPlugin(
                        skip_ego_id=False
                    ),
                    RenderVehicleToLaneletEdgesPlugin(
                        edge_arc=0.05,
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                    RenderTrafficGraphPlugin(
                        edge_color_other_connection=(0.0, 0.9, 0.0, 0.25),
                        edge_arc=0.0,
                        node_radius=0.65,
                        node_fillcolor=(0.0, 0.9, 0.0, 0.6),
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                ],
                caption="Geometric Viewer",
                **options
            )
        )
        renderers['traffic_graph'] = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                plugins=[
                    RenderTrafficGraphPlugin(
                        edge_color_other_connection=(0.0, 0.9, 0.0, 0.25),
                        edge_arc=0.0,
                        node_radius=0.65,
                        node_fillcolor=(0.0, 0.9, 0.0, 0.6),
                        temporal_alpha_multiplier=TEMPORAL_ALPHA_MULTIPLIER
                    ),
                ],
                caption="Traffic Graph Viewer",
                **options
            )
        )
        renderers['lanelet_graph'] = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                plugins=[
                    RenderLaneletGraphPlugin(
                        edge_arc=0.1
                    ),
                ],
                caption="Lanelet Graph Viewer",
                **options
            )
        )

    return renderers



def enjoy(args) -> None:

    if args.screenshots:
        shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    renderers = create_renderers(args)

    scenario_iterator = ScenarioIterator(
        directory=args.scenario_dir,
        save_scenario_pickles=True,
        load_scenario_pickles=True,
        preprocessors=PREPROCESSORS,
        prefilters=FILTERS,
        raise_exceptions=True
    )

    simulation_cls = SumoSimulation if args.interactive else ScenarioSimulation
    simulation_options_cls = partial(
        SumoSimulationOptions, 
        presimulation_steps=0
    ) if args.interactive else ScenarioSimulationOptions

    print(f"Enjoying {args.scenario_dir} ({len(scenario_iterator)} scenarios)")
    print(f"Preprocessing strategy: {PREPROCESSORS}")

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

        print(f"Enjoying {scenario_bundle.input_scenario_file}")
        
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
                data = extractor.extract(TrafficExtractionParams(
                    index=time_step,
                ))
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
                        output_file=f"{EXPORT_DIR}/enjoy_{frame_count}_{renderer_name}.png",
                        queued=True
                    )

            frame = simulation.render(
                renderers=list(renderers.values()),
                return_rgb_array=True,
                render_params=RenderParams(
                    time_step=time_step,
                    scenario=scenario,
                    data=data
                )
            )

            stdout(f"Enjoying {scenario.scenario_id} (timestep {time_step}/{simulation.final_time_step})")
            if args.max_timesteps is not None and time_step >= args.max_timesteps:
                break

            frame_count += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Play scenario videos.")
    parser.add_argument("--camera-follow", action="store_true", help="whether to let the camera follow a vehicle")
    parser.add_argument("--dt", type=float, help="time delta for simulation steps", default=0.04)
    parser.add_argument("--extract", action="store_true", help="whether to extract traffic graph data")
    parser.add_argument("--hd", action="store_true", help="high resolution rendering")
    parser.add_argument("--interactive", action="store_true", help="whether to simulate traffic using SUMO")
    parser.add_argument("--max-timesteps", type=int, help="optional maximum number of timesteps per scenario")
    parser.add_argument("--scenario-dir", type=str, default=INPUT_SCENARIO, help="path to scenario directory or scenario file")
    parser.add_argument("--screenshots", action="store_true", help="capture and export rendering screenshots")

    args = parser.parse_args()
    set_global_seed(0)
    enjoy(args)
