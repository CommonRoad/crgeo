import sys, os; sys.path.insert(0, os.getcwd())

import argparse
import shutil

from commonroad_geometric.common.logging import stdout
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractor, TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.iteration import ScenarioIterator
from commonroad_geometric.dataset.preprocessing.implementations.vehicle_filter_preprocessor import VehicleFilterPreprocessor
from commonroad_geometric.rendering.plugins.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.render_obstacles_plugin import RenderObstaclesPlugin, RenderObstaclesStyle
from commonroad_geometric.rendering.plugins.render_traffic_graph_plugin import RenderTrafficGraphPlugin, RenderTrafficGraphStyle
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions

# INPUT_SCENARIO = 'data/other/'
# INPUT_SCENARIO = 'data/t_junction_test'
# INPUT_SCENARIO = 'data/t_junction_recorded'
# INPUT_SCENARIO = 'data/other_recordings'
# INPUT_SCENARIO = 'data/munich_generated_dt_20'
# INPUT_SCENARIO = 'data/nuplan_sample_small_maps'
INPUT_SCENARIO = 'data/highway_test'
# INPUT_SCENARIO = 'data/osm_recordings'

PREPROCESSORS = [
    VehicleFilterPreprocessor(),
    #LaneletNetworkSubsetPreprocessor(radius=500.0),
    #DepopulateScenarioPreprocessor(5)
]

FILTERS = [
    # ValidTrajectoriesFilterer()
]

EXPORT_DIR = 'outputs/toolpaper/vtv'
EDGE_DRAWERS = [
    # VoronoiEdgeDrawer,
    # FullyConnectedEdgeDrawer,
    KNearestEdgeDrawer
]
EDGE_DISTANCE_THRESHOLD = 25.0
EXTRACT_TEMPORAL = True
RENDERER_SIZE = (400, 400)
SCREENSHOT_RATE = 1
HD_RESOLUTION_MULTIPLIER = 5.0
CAMERA_FOLLOW = False
VIEW_RANGE = 50.0
FPS = 30

LANELET_NETWORK_PLUGIN = RenderLaneletNetworkPlugin(
    lanelet_linewidth=0.85
)
OBSTACLES_PLUGIN = RenderObstaclesPlugin(
    RenderObstaclesStyle(
        #obstacle_fill_color=(0.2, 0.2, 0.2, 0.4),
        render_trail=False,
        fill_shadows=False
    )
)
V2V_TRAFFIC_GRAPH_PLUGIN = RenderTrafficGraphPlugin(
    RenderTrafficGraphStyle(
        edge_linewidth=1.9,
        edge_color_other_connection=(0.0, 0.9, 0.0, 0.5),
        node_radius=0.3,
        node_linewidth=0.01,
        node_fillcolor=(0.0, 1.0, 0.0, 1.0),
        render_temporal=False
    )
)
V2V_VTV_TRAFFIC_GRAPH_PLUGIN = RenderTrafficGraphPlugin(
    RenderTrafficGraphStyle(
        edge_linewidth=1.9,
        edge_color_other_connection=(0.0, 0.9, 0.0, 0.5),
        node_radius=0.3,
        node_linewidth=0.01,
        node_fillcolor=(0.0, 1.0, 0.0, 1.0),
        render_temporal=True
    )
)

PLUGIN_SETUPS = {
    'traffic': [LANELET_NETWORK_PLUGIN, OBSTACLES_PLUGIN],
    'v2v': [V2V_TRAFFIC_GRAPH_PLUGIN],
    'v2vvtv': [V2V_TRAFFIC_GRAPH_PLUGIN, V2V_VTV_TRAFFIC_GRAPH_PLUGIN],
    'all': [LANELET_NETWORK_PLUGIN, OBSTACLES_PLUGIN, V2V_TRAFFIC_GRAPH_PLUGIN, V2V_VTV_TRAFFIC_GRAPH_PLUGIN]
}


def generate_vtv_edges_plot(
    enable_screenshots: bool,
    enable_hd: bool,
    scenario_dir: str = INPUT_SCENARIO
) -> None:

    if enable_screenshots:
        shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    RENDERERS: dict[str, dict[str, TrafficSceneRenderer]] = {}
    for edge_drawer_cls in EDGE_DRAWERS:
        key = edge_drawer_cls.__name__
        RENDERERS[key] = {}
        for setup_name, setup in PLUGIN_SETUPS.items():
            renderer = TrafficSceneRenderer(
                options=TrafficSceneRendererOptions(
                    plugins=setup,
                    window_width=RENDERER_SIZE[0]*HD_RESOLUTION_MULTIPLIER if enable_hd else RENDERER_SIZE[0],
                    window_height=RENDERER_SIZE[1]*HD_RESOLUTION_MULTIPLIER if enable_hd else RENDERER_SIZE[1],
                    camera_follow=CAMERA_FOLLOW,
                    view_range=VIEW_RANGE,
                    fps=FPS,
                    transparent_screenshots=True,
                )
            )
            RENDERERS[key][setup_name] = renderer

    scenario_iterator = ScenarioIterator(
        directory=scenario_dir,
        save_scenario_pickles=True,
        load_scenario_pickles=True,
        preprocessors=PREPROCESSORS,
        prefilters=FILTERS,
        raise_exceptions=True
    )

    print(f"Enjoying {scenario_dir} ({len(scenario_iterator)} scenarios)")
    print(f"Preprocessing strategy: {PREPROCESSORS}")

    frame_count: int = 0
    for scenario_bundle in scenario_iterator:
        simulation = ScenarioSimulation(
            initial_scenario=scenario_bundle.preprocessed_scenario,
            options=ScenarioSimulationOptions(
                backup_current_scenario=False,
                backup_initial_scenario=False
            )
        )

        extractors = {}
        
        for edge_drawer_cls in EDGE_DRAWERS:
            if EXTRACT_TEMPORAL:
                sub_extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=edge_drawer_cls(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
                extractor = TemporalTrafficExtractor(
                    traffic_extractor=sub_extractor,
                    options=TemporalTrafficExtractorOptions(
                        collect_num_time_steps=15,
                        collect_skip_time_steps=0,
                        return_incomplete_temporal_graph=True
                    )
                )
            else:
                extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=edge_drawer_cls(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
            extractors[edge_drawer_cls.__name__] = extractor
        
        simulation.start()

        for time_step, scenario in simulation:

            for name, extractor in extractors.items():
                data = extractor.extract(TrafficExtractionParams(
                    index=time_step,
                ))

                for renderer_name, renderer in RENDERERS[name].items():
                    capture_screenshot = enable_screenshots and frame_count % SCREENSHOT_RATE == 0
                    if capture_screenshot:
                        renderer.screenshot(
                            output_file=f"{EXPORT_DIR}/vtv_edges_plot_{frame_count}_{name}_{renderer_name}.png",
                            queued=True
                        )

                    simulation.render(
                        renderers=renderer,
                        return_rgb_array=False,
                        render_params=RenderParams(
                            time_step=time_step,
                            scenario=scenario,
                            data=data
                        )
                    )

            stdout(f"Enjoying {scenario.scenario_id} (timestep {time_step}/{simulation.final_time_step})")

            frame_count += 1
        

