from commonroad_geometric.common.io_extensions.lanelet_network import remove_empty_intersections
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors import ScenarioPreprocessor


class RemoveEmptyIntersectionPreprocessor(ScenarioPreprocessor):
    def __init__(
        self,
    ) -> None:
        super(RemoveEmptyIntersectionPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        remove_empty_intersections(scenario_bundle.preprocessed_scenario.lanelet_network)
        return [scenario_bundle]


