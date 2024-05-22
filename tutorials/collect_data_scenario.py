import sys; import os; sys.path.insert(0, os.getcwd())

from pathlib import Path

from commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import DepopulateScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from commonroad_geometric.simulation.simulation_factory import SimulationFactory


# Tutorials demonstrate how CommonRoad-Geometric should be used.
# Do not modify this for your own purposes. Create a tool or project instead.
def collect_data_from_scenarios(
    scenario_dir: Path,
    preprocessor: ScenarioPreprocessor,
    samples_per_scenario: int,
    total_samples: int
) -> list[list[CommonRoadData]]:
    collector = DatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
            )
        ),
        simulation_factory=SimulationFactory(
            options=ScenarioSimulationOptions()  # We could specify options for the simulation here
        ),
        progress=True
    )

    scenario_iterator = ScenarioIterator(
        directory=scenario_dir,
        preprocessor=preprocessor,
        workers=2
    )

    dataset: list[list[CommonRoadData]] = []
    for scenario_bundle, _ in zip(scenario_iterator, range(total_samples)):
        scenario_bundle: ScenarioBundle
        print(f"Collecting data for {scenario_bundle.scenario_path}")
        scenario_data: list[CommonRoadData] = []
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=samples_per_scenario,
        ):
            scenario_data.append(data)
        dataset.append(scenario_data)

    return dataset


if __name__ == '__main__':
    dataset_ = collect_data_from_scenarios(
        scenario_dir=Path('data/highd-sample'),
        samples_per_scenario=100,
        total_samples=10,
        preprocessor=DepopulateScenarioPreprocessor(depopulator=5)
    )
    print(f"Collected {len(dataset_)} samples")
