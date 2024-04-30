from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor, T_ScenarioPreprocessorResult


class IdentityPreprocessor(BaseScenarioPreprocessor):
    """
    Every algebraic ring needs an identity. Does absolutely nothing.
    """

    @property
    def child_preprocessors(self) -> list[BaseScenarioPreprocessor]:
        return []

    @property
    def results_factor(self) -> int:
        return 1

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        return [scenario_bundle]
