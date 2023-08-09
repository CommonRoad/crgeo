from typing import Optional, Set, Tuple

from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class OffrouteCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        offroute = simulation.check_if_offroute()
        return offroute, 'Offroute' if offroute else None

    @property
    def reasons(self) -> Set[str]:
        return {'Offroute'}
