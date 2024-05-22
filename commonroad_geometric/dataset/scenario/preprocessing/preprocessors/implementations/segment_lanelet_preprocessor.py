from __future__ import annotations

from commonroad_geometric.common.io_extensions.lanelet_network import segment_lanelets
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import T_ScenarioPreprocessorResult
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork

class SegmentLaneletsPreprocessor(ScenarioPreprocessor):
    """
    The length of a lanelet is not bounded whereas lanelet node representations are
    of fixed size and as such can only capture a limited amount of information. To resolve this
    disparity we define an upper bound on the length of each lanelet and segment all lanelet
    which exceed the maximum length into a set of shorter lanelets before creating lanelet nodes.

    """

    def __init__(
        self,
        lanelet_max_segment_length: float = 50.0
    ) -> None:
        self.lanelet_max_segment_length = lanelet_max_segment_length
        super(SegmentLaneletsPreprocessor, self).__init__()

    def _process(self, scenario_bundle: ScenarioBundle) -> T_ScenarioPreprocessorResult:
        segmented_lanelet_network = segment_lanelets(
            scenario_bundle.preprocessed_scenario,
            lanelet_max_segment_length=self.lanelet_max_segment_length,
            validate=True
        )

        old_scenario = scenario_bundle.preprocessed_scenario
        new_scenario = Scenario(0.1, location=old_scenario.location, scenario_id=old_scenario.scenario_id)
        new_scenario.add_objects(segmented_lanelet_network)
        for obstacle in old_scenario.dynamic_obstacles:
            obstacle.initial_shape_lanelet_ids = None
            obstacle.prediction.shape_lanelet_assignment = None
            new_scenario.add_objects(obstacle)

        scenario_bundle.preprocessed_scenario = new_scenario
        return [scenario_bundle]
