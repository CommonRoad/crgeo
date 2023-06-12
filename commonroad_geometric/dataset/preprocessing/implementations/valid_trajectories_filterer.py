from commonroad.scenario.scenario import Scenario
from commonroad_geometric.dataset.preprocessing.base_scenario_filterer import BaseScenarioFilterer
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object


class ValidTrajectoriesFilterer(BaseScenarioFilterer):

    def __init__(
        self
    ):
        super().__init__()

    def _has_overlap(self, scenario: Scenario) -> bool:
        collision_checkers = dict(
            (obs.obstacle_id, create_collision_object(obs)) for obs in scenario.dynamic_obstacles
        )
        for obs_outer in scenario.dynamic_obstacles:
            cc_outer = collision_checkers[obs_outer.obstacle_id]
            for obs_inner in scenario.dynamic_obstacles:
                if obs_inner.obstacle_id == obs_outer.obstacle_id:
                    continue
                cc_inner = collision_checkers[obs_inner.obstacle_id]
                res = cc_outer.collide(cc_inner)
                if res:
                    return True
        return False

    def _filter_scenario(self, scenario: Scenario) -> bool:
        has_overlap = self._has_overlap(scenario)
        return not has_overlap
