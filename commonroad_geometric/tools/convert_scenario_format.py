import argparse
from pathlib import Path

from commonroad.common.file_writer import OverwriteExistingFile

from commonroad_geometric.common.io_extensions.scenario_file_format import ScenarioFileFormat
from commonroad_geometric.common.progress_reporter import ProgressReporter
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.scenario_format_converter import ScenarioFormatConverter


def convert_scenario_format(
    scenario_dir: Path,
    input_format: ScenarioFileFormat,
    output_dir: Path,
    output_format: ScenarioFileFormat,
    overwrite_output_dir: bool,
    num_workers: int,
):
    overwrite_existing_file = OverwriteExistingFile.ASK_USER_INPUT
    if overwrite_output_dir:
        overwrite_existing_file = OverwriteExistingFile.ALWAYS
    scenario_iterator = ScenarioIterator(
        directory=scenario_dir,
        file_format=input_format,
        preprocessor=ScenarioFormatConverter(
            output_directory=output_dir,
            output_file_format=output_format,
            overwrite_existing_file=overwrite_existing_file
        ),
        workers=num_workers,
    )

    total = scenario_iterator.max_result_scenarios
    progress_reporter = ProgressReporter(initial=0, total=total)
    for index, scenario_bundle in enumerate(scenario_iterator):
        progress_reporter.update(index + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert scenarios from one file format to another format.")
    parser.add_argument("--scenario-dir", type=Path,
                        help="path to scenario directory which traffic should be generated")
    parser.add_argument("--input-format", choices=['xml', 'bundle', 'all'], default='xml',
                        help="output directory for the scenarios with the generated traffic")
    parser.add_argument("--output-dir", type=Path,
                        help="output directory for the scenarios with the generated traffic")
    parser.add_argument("--output-format", choices=['xml', 'bundle', 'all'], default='xml',
                        help="output directory for the scenarios with the generated traffic")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove and re-create the output directory before traffic generation")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of parallel workers generating traffic")
    args = parser.parse_args()

    suffix_to_format: dict[str, ScenarioFileFormat] = dict(
        xml=ScenarioFileFormat.XML,
        bundle=ScenarioFileFormat.BUNDLE,
        all=ScenarioFileFormat.ALL
    )
    _input_format = suffix_to_format[args.input_format]
    _output_format = suffix_to_format[args.output_format]

    convert_scenario_format(
        scenario_dir=args.scenario_dir,
        input_format=_input_format,
        output_dir=args.output_dir,
        output_format=_output_format,
        overwrite_output_dir=args.overwrite,
        num_workers=args.num_workers,
    )
