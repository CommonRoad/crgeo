import logging
from pathlib import Path
from typing import Optional

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from commonroad_geometric.common.io_extensions.scenario_files import ScenarioFileFormat
from commonroad_geometric.common.utils.filesystem import FileFormatNotSupportedError, save_dill
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors import ScenarioPreprocessor

logger = logging.getLogger(__name__)


class ScenarioFormatConverter(ScenarioPreprocessor):
    """
    This preprocessor saves the preprocessed scenario and planning problems set to the specified format.
    """

    def __init__(
        self,
        output_file_format: ScenarioFileFormat,
        output_directory: Optional[Path] = None,
        decimal_precision: int = 10,
        overwrite_existing_file: OverwriteExistingFile = OverwriteExistingFile.ALWAYS
    ):
        """
        Args:
            output_file_format (ScenarioFileFormat): Output file format
            output_directory (Optional[Path]): Optional path to save the pickles to
            decimal_precision (int): Decimal precision for CommonRoadFileWriter. Defaults to 10 (instead of 4).
            overwrite_existing_file (OverwriteExistingFile): Whether to overwrite an existing file. Defaults to always.
        """
        self._output_file_format = output_file_format
        self._output_directory = output_directory
        self._decimal_precision = decimal_precision
        self._overwrite_existing_file = overwrite_existing_file
        super(ScenarioFormatConverter, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        """
        Args:
            scenario_bundle (ScenarioBundle): the scenario bundle which will be saved in the specified file format

        Returns:
            the unmodified scenario_bundle
        """
        filename_without_suffix = scenario_bundle.scenario_path.stem
        if self._output_directory is not None:
            output_path = self._output_directory / filename_without_suffix
        else:
            output_path = scenario_bundle.scenario_path.parent / filename_without_suffix

        for suffix in self._output_file_format.suffixes:
            output_path = output_path.with_suffix(suffix)
            match suffix:
                case '.pkl':
                    if self._overwrite_existing_file == OverwriteExistingFile.ALWAYS or not output_path.is_file():
                        cached_path = scenario_bundle.scenario_path
                        # Change path during file writing (has temporary side effect)
                        scenario_bundle.scenario_path = output_path
                        save_dill(scenario_bundle, file_path=output_path)
                        # Revert path as to not modify original bundle
                        scenario_bundle.scenario_path = cached_path
                    else:
                        logger.warning(f"Overwriting file: {output_path}")
                case '.xml':
                    writer = CommonRoadFileWriter(
                        scenario=scenario_bundle.preprocessed_scenario,
                        planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
                        decimal_precision=self._decimal_precision,
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer.write_to_file(
                        filename=str(output_path),
                        overwrite_existing_file=self._overwrite_existing_file
                    )
                case _format:
                    raise FileFormatNotSupportedError(f"File format {_format} is not supported. {self._name} supports "
                                                      f"{{'.xml', '.pkl'}}")
        return [scenario_bundle]
