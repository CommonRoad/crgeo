import os
import sys

sys.path.insert(0, os.getcwd())

from typing import List

from commonroad_geometric.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.iteration import ScenarioIterator
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor

SCENARIO_DIR = 'data/osm_recordings'
SAMPLES_PER_SCENARIO = 100
TOTAL_SAMPLES = 10
SCENARIO_PREPROCESSORS: List[BaseScenarioPreprocessor] = [
    #(DepopulateScenarioPreprocessor(5), 3),
]


def get_dataset() -> List[List[CommonRoadData]]:
    collector = ScenarioDatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
            )
        )
    )
    dataset: List[List[CommonRoadData]] = [[]]
    iterator = ScenarioIterator(
        SCENARIO_DIR,
        preprocessors=SCENARIO_PREPROCESSORS,
        load_scenario_pickles=False,
        save_scenario_pickles=False,
        verbose=1
    )
    counter = 0
    for scenario_bundle in iterator:
        
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=SAMPLES_PER_SCENARIO
        ):
            dataset[-1].append(data)
            if len(dataset) >= TOTAL_SAMPLES:
                return dataset
        dataset.append([])
        counter += 1
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
    print(f"Collected {len(dataset)} samples")
