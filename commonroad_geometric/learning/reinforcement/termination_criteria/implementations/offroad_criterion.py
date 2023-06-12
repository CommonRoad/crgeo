from typing import Optional, Set, Tuple

from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class OffroadCriterion(BaseTerminationCriterion):
    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        offroad = simulation.check_if_offroad()
        return offroad, 'Offroad' if offroad else None

    @property
    def reasons(self) -> Set[str]:
        return {'Offroad'}
