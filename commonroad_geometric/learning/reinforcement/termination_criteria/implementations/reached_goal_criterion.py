from typing import Optional, Set, Tuple

from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class ReachedGoalCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        reached_goal = simulation.check_if_has_reached_goal()
        return reached_goal, 'ReachedGoal' if reached_goal else None

    @property
    def reasons(self) -> Set[str]:
        return {'ReachedGoal'}


class OvershotGoalCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        overshot_goal = simulation.check_if_overshot_goal()
        return overshot_goal, 'OvershotGoal' if overshot_goal else None

    @property
    def reasons(self) -> Set[str]:
        return {'OvershotGoal'}
