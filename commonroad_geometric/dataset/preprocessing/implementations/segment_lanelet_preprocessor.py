from __future__ import annotations
from typing import Optional, Tuple

from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.common.io_extensions.lanelet_network import segment_lanelets
from commonroad.planning.planning_problem import PlanningProblemSet


class SegmentLaneletsPreprocessor(BaseScenarioPreprocessor):
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

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:

        lanelet_network = segment_lanelets(
            scenario,
            lanelet_max_segment_length=self.lanelet_max_segment_length,
            validate=True
        )
        scenario.lanelet_network = lanelet_network
        return scenario, planning_problem_set
