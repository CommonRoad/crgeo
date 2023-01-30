import os
import sys

sys.path.insert(0, os.getcwd())

from typing import List

from crgeo.dataset.collection.scenario_dataset_collector import ScenarioDatasetCollector
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from crgeo.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from crgeo.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, TrafficExtractorFactory
from crgeo.dataset.iteration import ScenarioIterator
from crgeo.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor

#SCENARIO_DIR = '.'
SCENARIO_DIR = 'data/osm_recordings'
SAMPLES_PER_SCENARIO = 100
TOTAL_SAMPLES = 10
SCENARIO_PREPROCESSORS: List[BaseScenarioPreprocessor] = [
    #(DepopulateScenarioPreprocessor(5), 3),
]

def get_dataset() -> List[List[CommonRoadData]]:
    collector = ScenarioDatasetCollector(
        extractor_factory=TemporalTrafficExtractorFactory(
            traffic_extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    # render = False,
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50)
            )),
            options=TemporalTrafficExtractorOptions(
                collect_num_time_steps=40,
                return_incomplete_temporal_graph=True
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
    counter=0
    for scenario_bundle in iterator:

        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=SAMPLES_PER_SCENARIO
        ):
            # print(time_step, ", ".join([f"{elm:.3f}" for elm in data.v.velocity[:, 0].tolist()]))
            if time_step % 10 == 0:
                # Getting vehicle velocity history tensor
                velocity_history = data.get_node_features_temporal_sequence(['velocity']) # TODO: Wrong type hint CommonRoadData

                # Getting vehicle velocity time series for a single vehicle
                vehicle_id = data.v.id[0].item()
                velocity_history_single_vehicle = data.get_node_features_temporal_sequence_for_vehicle(vehicle_id, ['velocity'])

            # Storing temporal data object
            dataset[-1].append(data)
            if len(dataset) >= TOTAL_SAMPLES:
                return dataset
        dataset.append([])
        counter+=1
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
    print(f"Collected {len(dataset)} samples")
