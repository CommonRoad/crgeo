from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, List
import logging

from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.io_extensions.lanelet_network import merge_successors
from commonroad_geometric.dataset.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad.planning.planning_problem import PlanningProblemSet

logger = logging.getLogger(__name__)


class MergeLaneletsPreprocessor(BaseScenarioPreprocessor):
    """
    Merges lanelets that can be merged.
    """

    def __init__(
        self, 
    ) -> None:
        super(MergeLaneletsPreprocessor, self).__init__()

    def _process(
        self,
        scenario: Scenario,
        planning_problem_set: Optional[PlanningProblemSet]
    ) -> Tuple[Scenario, Optional[PlanningProblemSet]]:
        
        try:
            merge_successors(scenario.lanelet_network, validate=True)
        except Exception as e:
            logger.error(e, exc_info=True)

        return scenario, planning_problem_set
