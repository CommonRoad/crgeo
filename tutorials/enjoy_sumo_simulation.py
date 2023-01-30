import sys, os; sys.path.insert(0, os.getcwd())

import logging

from crgeo.common.logging import setup_logging
from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from crgeo.rendering.plugins import RenderLaneletNetworkPlugin, RenderObstaclesPlugin
from crgeo.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from crgeo.rendering.types import RenderParams
from crgeo.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from crgeo.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantRateSpawner

PRESIMULATION_STEPS = 0 # "auto" 
#INPUT_SCENARIO = 'data/other/USA_Peach-1_1_T-1.xml'
#INPUT_SCENARIO = 'data/t_junction_test/ZAM_Tjunction-1_4_T-1.xml'
#INPUT_SCENARIO = 'data/osm_crawled/DEU_Berlin_2-2341.xml'
INPUT_SCENARIO = 'data/other/FRA_Miramas-4_8_T-1.xml'
TIME_STEPS = 500


def enjoy() -> None:
    setup_logging(
        level=logging.DEBUG,
    )

    traffic_spawner = ConstantRateSpawner(
        p_spawn=0.005,
        max_vehicles=15
    )

    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=SumoSimulationOptions(
            presimulation_steps=PRESIMULATION_STEPS,
            traffic_spawner=traffic_spawner
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
            RenderObstaclesPlugin(),
        ],
    ))
    simulation.start()

    render_kwargs = dict(randomize_color=False)
    for time_step, scenario in simulation:
        data = extractor.extract(TrafficExtractionParams(
            index=time_step
        ))
        simulation.render(
            renderers=[renderer],
            render_params=RenderParams(render_kwargs=render_kwargs)
        )


if __name__ == '__main__':
    enjoy()
