from typing import Set, Tuple, Optional
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class CollisionCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        collision_struct = ego_vehicle_simulation.check_if_has_collision()
        if collision_struct.collision:
            return True, 'Collision' if collision_struct.ego_at_fault else 'CollisionNonFault'
        return False, None

    @property
    def reasons(self) -> Set[str]:
        return {'Collision', 'CollisionNonFault'}