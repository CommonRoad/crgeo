import sys
import os; sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.implementations import RenderVehicleToLaneletEdgesPlugin
from commonroad_geometric.rendering.plugins.implementations.render_lanelet_network_plugin import RenderLaneletNetworkPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantRateSpawner

PRESIMULATION_STEPS = 0  # "auto"
# INPUT_SCENARIO = Path('data/other/USA_Peach-1_1_T-1.xml')
# INPUT_SCENARIO = Path('data/t_junction_test/ZAM_Tjunction-1_4_T-1.xml')
# INPUT_SCENARIO = Path('data/osm_crawled/DEU_Berlin_2-2341.xml')
INPUT_SCENARIO = Path('data/other/FRA_Miramas-4_8_T-1.xml')
TIME_STEPS = 500


def enjoy(max_timesteps=100) -> None:
    setup_logging(
        level=logging.DEBUG,
    )

    traffic_spawner = ConstantRateSpawner(
        p_spawn=0.05,
        max_vehicles=15
    )

    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=SumoSimulationOptions(
            presimulation_steps=PRESIMULATION_STEPS,
            traffic_spawner=traffic_spawner,
            ignore_assignment_opposite_direction=False
        )
    )

    extractor = TrafficExtractor(
        simulation=simulation,
        options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer(dist_threshold=20),
        )
    )

    renderer = TrafficSceneRenderer(TrafficSceneRendererOptions(
        plugins=[
            # RenderLaneletGraphPlugin(),
            RenderLaneletNetworkPlugin(),
            RenderObstaclePlugin(obstacle_color=Color('green')),
            # RenderVehicleToLaneletEdgesPlugin()
        ],
        # viewer_options=Open3DViewerOptions()
        viewer_options=GLViewerOptions(),
    ))
    simulation.start()

    render_kwargs = dict(randomize_color=False)
    for time_step, scenario in simulation:
        data = extractor.extract(
            time_step=time_step
        )
        simulation.render(
            renderers=[renderer],
            render_params=RenderParams(
                data=data,
                render_kwargs=render_kwargs
            )
        )
        if max_timesteps is not None and time_step >= max_timesteps:
            break


if __name__ == '__main__':
    enjoy(max_timesteps=1000)
