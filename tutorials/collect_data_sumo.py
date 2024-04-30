import sys; import os; sys.path.insert(0, os.getcwd())

from pathlib import Path

from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin, RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulationOptions
from commonroad_geometric.simulation.simulation_factory import SimulationFactory


# Tutorials demonstrate how CommonRoad-Geometric should be used.
# Do not modify this for your own purposes. Create a tool or project instead.
def collect_data_from_sumo(
    scenario_path: Path,
    samples_per_scenario: int,
) -> list[list[CommonRoadData]]:
    collector = DatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=25),
            )
        ),
        simulation_factory=SimulationFactory(
            options=SumoSimulationOptions(  # Using SUMO to simulate traffic is as easy as specifying the options
                presimulation_steps=500
            )
        ),
        progress=True
    )

    scenario_iterator = ScenarioIterator(
        # ScenarioIterator will automagically figure out that this is one scenario
        directory=scenario_path,
        workers=1,
    )

    renderer = TrafficSceneRenderer(
        options=TrafficSceneRendererOptions(
            plugins=[
                RenderLaneletNetworkPlugin(),
                RenderTrafficGraphPlugin(render_temporal=False),
                RenderObstaclePlugin(),
            ]
        )
    )
    dataset: list[list[CommonRoadData]] = []
    for scenario_bundle in scenario_iterator:
        print(f"Collecting data for {scenario_bundle.scenario_path}")
        scenario_data: list[CommonRoadData] = []
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=samples_per_scenario,
        ):
            scenario_data.append(data)
            # We can render the simulated traffic scene from CommonRoadData when setting from_graph=True
            frame = renderer.render(
                render_params=RenderParams(
                    time_step=time_step,
                    scenario=scenario_bundle.preprocessed_scenario,  # Required to render lanelet network
                    data=data,
                    render_kwargs=dict(
                        # Automagically overwrite attribute of RenderLaneletNetworkPlugin and RenderObstaclePlugin
                        from_graph=True,
                        overlays={'Timestep': time_step}
                    )
                ),
                return_frame=False
            )
            assert frame is None
        dataset.append(scenario_data)

    return dataset


if __name__ == '__main__':
    dataset_ = collect_data_from_sumo(
        scenario_path=Path('data/osm_crawled/DEU_Munich_1-100.xml'),
        samples_per_scenario=300,
    )
    print(f"Collected {len(dataset_)} samples")
