import sys, os; sys.path.insert(0, os.getcwd())

import shutil
from typing import Iterable

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from torch_geometric.loader import DataLoader

from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.iteration.timestep_iterator import TimeStepIterator
from commonroad_geometric.debugging.profiling import profile
from commonroad_geometric.rendering.plugins import RenderLaneletNetworkPlugin, RenderObstaclesPlugin, RenderTrafficGraphPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions

DATASET_DIR = 'tutorials/output/render_scenario_from_graph/dataset'
SCENARIO_DIR = 'data/osm_recordings'
FIGURE_DIR = 'tutorials/output/render_scenario_from_graph/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)
DATA_COLLECTOR_CLS = ScenarioDatasetCollector
SAVE_GRAPH_DATA_SNAPSHOT = False
MAX_SAMPLES = 10
NUM_WORKERS = 1
MAX_SCENARIOS = 1


def pre_transform(scenario: Scenario, planning_problem_set: PlanningProblemSet) -> Iterable[CommonRoadData]:
    extractor_factory = TrafficExtractorFactory(TrafficExtractorOptions(
        edge_drawer=VoronoiEdgeDrawer()
    ))
    collector = ScenarioDatasetCollector(
        extractor_factory=extractor_factory,
        simulation_options=ScenarioSimulationOptions(
            collision_checking=False,
            step_renderers=[TrafficSceneRenderer()]
        )
    )

    for time_step, data in collector.collect(
        scenario,
        planning_problem_set=planning_problem_set,
        max_samples=MAX_SAMPLES,
        progress=True
    ):
        yield data

def main() -> None:
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    dataset: CommonRoadDataset[CommonRoadData, CommonRoadData] = CommonRoadDataset(
        raw_dir=SCENARIO_DIR,
        processed_dir=DATASET_DIR,
        pre_transform=pre_transform,
        pre_transform_workers=NUM_WORKERS,
        max_scenarios=MAX_SCENARIOS
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for index, batch in enumerate(loader):
        scenario, _ = CommonRoadFileReader(next(x for x in dataset.raw_paths if batch.scenario_id[0] in x)).open()
        
        # Load scenario
        timestep_iterator = TimeStepIterator(scenario, loop=True)

        renderer_plugins = [
            RenderObstaclesPlugin(
                font_size=5,
                draw_index=True,
            ),
            RenderTrafficGraphPlugin(),
            RenderLaneletNetworkPlugin()
        ]

        renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                window_height=1080,
                window_width=1920,
                plugins=renderer_plugins,
            )
        )

        frames = renderer.render(
            return_rgb_array=True,
            render_params=RenderParams(
                scenario=timestep_iterator.scenario,
                time_step=0,
                data=batch
            ),
        )

        if SAVE_GRAPH_DATA_SNAPSHOT:
            from pyglet.image import ImageData
            assert isinstance(frames, ImageData)
            file_name = os.path.join(FIGURE_DIR, f'generate_scenario_from_graph_snapshot_{index}.png')
            frames.save(file_name)
            print(f"Saved image to {file_name}")

        renderer.close()


if __name__ == '__main__':
    profile(main)
