from typing import Optional, Set, Tuple

from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class FrictionViolationCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        violates_friction = simulation.check_if_violates_friction()
        return violates_friction, 'FrictionViolation' if violates_friction else None

    @property
    def reasons(self) -> Set[str]:
        return {'FrictionViolation'}
