import sys, os; sys.path.insert(0, os.getcwd())

import argparse
from functools import partial
from pathlib import Path

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from commonroad_geometric.common.io_extensions.scenario_files import filter_scenario_paths, find_scenario_paths
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations import OverlappingTrajectoriesFilter

SCENARIO_DIR = 'data/osm_recordings'
OUTPUT_DIR = Path('tutorials/output/osm_recordings_processed')
SKIP_EXISTING = True


def process(args) -> None:
    if SKIP_EXISTING:
        skip_scenarios = find_scenario_paths(args.output_dir)
        print(f"Skipping {len(skip_scenarios)} scenarios")
    else:
        skip_scenarios = None

    preprocessor = OverlappingTrajectoriesFilter()
    # preprocessor >>= HighwayFilter()

    # preprocessor >>= SegmentLaneletPreprocessor()
    # preprocessor >>= LaneletNetworkSubsetPreprocessor()
    # preprocessor >>= RemoveIslandsPreprocessor()
    # depopulation_preprocessor = DepopulateScenarioPreprocessor(5)
    # preprocessor >>= (depopulation_preprocessor | depopulation_preprocessor | depopulation_preprocessor)

    # preprocessor >>= MinIntersectionFilter(min_intersections=1)

    iterator = ScenarioIterator(
        directory=Path(args.scenario_dir),
        filter_scenario_paths=partial(filter_scenario_paths,
                                      max_scenarios=args.max_scenarios,
                                      excluded_scenario_names=skip_scenarios),
        preprocessor=preprocessor,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Created scenario iterator of length {iterator.max_result_scenarios}")

    for scenario_bundle in iterator:
        output_path = os.path.join(args.output_dir, scenario_bundle.scenario_path.name)
        print(f"Processing scenario {scenario_bundle.scenario_path}")
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
