from typing import Set, Tuple, Optional
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class TimeoutCriterion(BaseTerminationCriterion):
    def __init__(self, max_timesteps: int) -> None:
        assert max_timesteps is None or max_timesteps > 0
        self._max_timesteps = max_timesteps
        super().__init__()

    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        if self._max_timesteps is None:
            return False, None
        if simulation.simulation.current_run_start is None:
            return False, None
        is_timeout = simulation.current_time_step - simulation.simulation.current_run_start >= self._max_timesteps
        return is_timeout, 'Timeout' if is_timeout else None

    @property
    def reasons(self) -> Set[str]:
        return {'Timeout'}
