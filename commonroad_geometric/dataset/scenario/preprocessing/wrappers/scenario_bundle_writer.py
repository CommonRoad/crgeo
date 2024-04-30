from pathlib import Path
from typing import Optional

from commonroad_geometric.common.utils.filesystem import save_dill
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors import ScenarioPreprocessor


class ScenarioBundleWriter(ScenarioPreprocessor):
    """
    This preprocessor pickles the scenario bundle it receives to a file with the extension '.pkl'.

    Uses a hash as part of the filename to avoid collisions with other scenario bundles
    derived from the same scenario path.
    This is preferred over a UUID here, as we want identical scenario bundles to collide.

    Closely related to ScenarioFormatConverter preprocessor.
    """

    def __init__(self, output_directory: Optional[Path] = None):
        """
        Args:
            output_directory (Optional[Path]): Optional path to save the pickles to
        """
        self._output_directory = output_directory
        super(ScenarioBundleWriter, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        """
        Args:
            scenario_bundle (ScenarioBundle): the scenario bundle which will be pickled and saved to a file

        Returns:
            the unmodified scenario_bundle
        """
        # Use hash as part of filename to avoid collisions with other scenario bundles
        filename_without_suffix = f"{scenario_bundle.scenario_path.stem}_{hash(scenario_bundle)}"

        if self._output_directory is not None:
            pickle_export_path = self._output_directory / filename_without_suffix
        else:
            pickle_export_path = scenario_bundle.scenario_path.parent / filename_without_suffix
        pickle_export_path = pickle_export_path.with_suffix('.pkl')

        if not pickle_export_path.is_file():
            cached_path = scenario_bundle.scenario_path
            # Change path during file writing (has temporary side-effect)
            scenario_bundle.scenario_path = pickle_export_path
            save_dill(scenario_bundle, file_path=pickle_export_path)
            # Revert path as to not modify original bundle
            scenario_bundle.scenario_path = cached_path
        return [scenario_bundle]
