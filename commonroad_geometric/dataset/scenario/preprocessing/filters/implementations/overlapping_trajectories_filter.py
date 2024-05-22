from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object

from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.preprocessing.filters.scenario_filter import ScenarioFilter


def _has_trajectory_overlap(scenario: Scenario) -> bool:
    obstacle_id_to_collision_checker = {obs.obstacle_id: create_collision_object(obs) for obs in scenario.dynamic_obstacles}
    for obs_outer in scenario.dynamic_obstacles:
        cc_outer = obstacle_id_to_collision_checker[obs_outer.obstacle_id]
        for obs_inner in scenario.dynamic_obstacles:
            if obs_inner.obstacle_id == obs_outer.obstacle_id:
                continue
            cc_inner = obstacle_id_to_collision_checker[obs_inner.obstacle_id]
            res = cc_outer.collide(cc_inner)
            if res:
                return True
    return False


class OverlappingTrajectoriesFilter(ScenarioFilter):
    """
    Rejects scenarios where the paths of trajectories overlap, i.e. where the obstacles collide.
    """

    def _filter(self, scenario_bundle: ScenarioBundle) -> bool:
        has_trajectory_overlap = _has_trajectory_overlap(scenario_bundle.preprocessed_scenario)
        return not has_trajectory_overlap
