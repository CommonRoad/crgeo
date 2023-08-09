import sys, os; sys.path.insert(0, os.getcwd())

import logging
import argparse

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad_geometric.common.io_extensions.scenario import find_scenario_files
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.dataset.iteration import ScenarioIterator
from commonroad_geometric.dataset.preprocessing.implementations.depopulate_scenario_preprocessor import DepopulateScenarioPreprocessor
from commonroad_geometric.dataset.preprocessing.implementations.intersection_filterer import IntersectionFilterer
from commonroad_geometric.dataset.preprocessing.implementations.lanelet_network_subset_preprocessor import LaneletNetworkSubsetPreprocessor
from commonroad_geometric.dataset.preprocessing.implementations.multilane_filterer import MultiLaneFilterer
from commonroad_geometric.dataset.preprocessing.implementations.remove_islands_preprocessor import RemoveIslandsPreprocessor
from commonroad_geometric.dataset.preprocessing.implementations.segment_lanelet_preprocessor import SegmentLaneletsPreprocessor
from commonroad_geometric.dataset.preprocessing.implementations.valid_trajectories_filterer import ValidTrajectoriesFilterer
from commonroad_geometric.debugging.profiling import profile

SCENARIO_DIR = 'data/osm_recordings'
OUTPUT_DIR = 'tutorials/output/osm_recordings_processed'
SCENARIO_PREPROCESSORS = [
    #SegmentLaneletsPreprocessor(),
    #(LaneletNetworkSubsetPreprocessor(), 1),
    #RemoveIslandsPreprocessor(),
    #(DepopulateScenarioPreprocessor(5), 3),
    #FixLaneletNetworkPreprocessor()
]
SCENARIO_PREFILTERS = [
    ValidTrajectoriesFilterer(),
    #MultiLaneFilterer(keep_multilane=False),
]
SCENARIO_POSTFILTERS = [
    #IntersectionFilterer(min_intersections=1)
]
SKIP_EXISTING = True

def process(args) -> None:
    if SKIP_EXISTING:
        skip_scenarios = find_scenario_files(args.output_dir)
        print(f"Skipping {len(skip_scenarios)} scenarios")
    else:
        skip_scenarios = None

    iterator = ScenarioIterator(
        args.scenario_dir,
        preprocessors=SCENARIO_PREPROCESSORS,
        prefilters=SCENARIO_PREFILTERS,
        postfilters=SCENARIO_POSTFILTERS,
        load_scenario_pickles=True,
        save_scenario_pickles=False,
        verbose=1,
        max_scenarios=args.max_scenarios,
        skip_scenarios=skip_scenarios
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Created scenario iterator of length {len(iterator)}")

    for scenario_bundle in iterator:
        output_path = os.path.join(args.output_dir, scenario_bundle.input_scenario_file.name)
        print(f"Processing scenario {scenario_bundle.input_scenario_file}")
        file_writer = CommonRoadFileWriter(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set
        )
        file_writer.write_to_file(
            str(output_path),
            overwrite_existing_file=OverwriteExistingFile.ALWAYS,
            check_validity=False
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process and filter scenarios.")
    parser.add_argument("--scenario-dir", type=str, default=SCENARIO_DIR, help="path to scenario directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--max-scenarios", type=int, help="max scenarios to process")

    args = parser.parse_args()
    process(args)

