import logging
from pathlib import Path

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult

logger = logging.getLogger(__name__)


def preprocess_scenario(
    scenario_path: Path,
    preprocessor: BaseScenarioPreprocessor,
) -> T_ScenarioPreprocessorResult:
    logger.debug(f"Preprocessing scenario_path={scenario_path} with preprocessor={preprocessor.name}")
    scenario_bundle = ScenarioBundle.read(
        scenario_path=scenario_path,
        lanelet_assignment=False
    )
    processing_results = preprocessor(scenario_bundle)
    return processing_results
