import sys, os; sys.path.insert(0, os.getcwd())

import torch_geometric.loader

from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor, TrafficExtractorOptions
from crgeo.rendering.traffic_scene_renderer import TrafficSceneRenderer
from crgeo.rendering.types import RenderParams
from crgeo.rendering.video_recording import save_video_from_frames
from crgeo.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions

INPUT_SCENARIO = 'data/osm_crawled/DEU_Munich_1-100.xml'
# INPUT_SCENARIO = 'data/other/USA_US101-26_1_T-1.xml'
OUTPUT_SCENARIO = 'tutorials/output/sumo_sim.gif'

PRESIMULATION_STEPS = 0
NUM_TIMESTEPS = 300


def collect_data() -> None:
    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=SumoSimulationOptions(
            presimulation_steps=PRESIMULATION_STEPS
        )
    )

    traffic_extractor = TrafficExtractor(
        simulation=simulation,
        options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer(dist_threshold=25),
        )
    )

    renderer = TrafficSceneRenderer()
    frames = []
    samples = []

    traffic_extractor.simulation.start()

    for time_step, scenario in traffic_extractor.simulation(to_time_step=NUM_TIMESTEPS):
        data = traffic_extractor.extract(TrafficExtractionParams(
            index=time_step,
        ))
        frame = simulation.render(
            renderers=[renderer],
            return_rgb_array=True,
            render_params=RenderParams(data=data),
            overlays={
                'Timestep': time_step
            }
        )
        samples.append(data)
        frames.append(frame)

    traffic_extractor.simulation.close()

    save_video_from_frames(frames, OUTPUT_SCENARIO)

    data_loader = torch_geometric.loader.DataLoader(
        samples,
        batch_size=10,
        shuffle=True
    )
    print(data_loader)


if __name__ == '__main__':
    collect_data()
