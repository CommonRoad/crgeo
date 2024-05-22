from __future__ import annotations

import logging

from commonroad_geometric.common.io_extensions.lanelet_network import merge_successors
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor

logger = logging.getLogger(__name__)


class MergeLaneletsPreprocessor(ScenarioPreprocessor):
    """
    Merges lanelets that can be merged.
    """

    def __init__(
        self, 
    ) -> None:
        super(MergeLaneletsPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        try:
            merge_successors(scenario_bundle.preprocessed_scenario.lanelet_network, validate=True)
        except Exception as e:
            logger.error(e, exc_info=True)

        return [scenario_bundle]
