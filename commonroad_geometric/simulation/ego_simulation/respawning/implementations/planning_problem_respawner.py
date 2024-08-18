from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, cast

from commonroad.geometry.shape import ShapeGroup

from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import BaseRespawner, BaseRespawnerOptions, RespawnerSetupFailure, T_Respawn_Tuple

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


@dataclass
class PlanningProblemRespawnerOptions(BaseRespawnerOptions):
    ...


class PlanningProblemRespawner(BaseRespawner):
    """
    Respawns ego vehicle according to the planning problem associated with the scenario.
    """
    # TODO: probably broken?

    def __init__(
        self,
        options: Optional[PlanningProblemRespawnerOptions] = None
    ) -> None:
        options = options or PlanningProblemRespawnerOptions()
        if isinstance(options, dict):
            options = PlanningProblemRespawnerOptions(**options)
        self._options: PlanningProblemRespawnerOptions = options
        super().__init__(options=options)

    def _setup(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
        planning_problem = ego_vehicle_simulation.planning_problem
        if planning_problem is None:
            raise RespawnerSetupFailure(
                f"PlanningProblemRespawner needs a planning problem, which it appears is lacking for the current scenario {ego_vehicle_simulation.current_scenario.scenario_id}. Consider using the RandomRespawner instead.")
        self._initial_state = planning_problem.initial_state
        goal_state = planning_problem.goal.state_list[0]
        self._goal_state = goal_state.shapes[0] if isinstance(goal_state, ShapeGroup) else goal_state
        self._goal_position = self._goal_state.position.shapes[0] if isinstance(
            self._goal_state.position, ShapeGroup) else self._goal_state.position  # Note: position itself is a shape

        lanelet_network = ego_vehicle_simulation.current_scenario.lanelet_network
        goal_lanelet_assignment_position = ego_vehicle_simulation.current_scenario.lanelet_network.find_lanelet_by_position([
                                                                                                                            self._goal_position.center])

        goal_lanelet_id = None
        # Fallback if assignment by position fails, happens in edge case where ego
        # vehicle exits into missing lanelet or in crowded intersection
        if goal_lanelet_assignment_position == [[]]:
            lanelet_assignment_shape = ego_vehicle_simulation.current_scenario.lanelet_network.find_lanelet_by_shape(
                self._goal_position)
            # only 1-dimensional list
            if len(lanelet_assignment_shape) > 1:
                # just guessing
                exit_candidates = {lanelet.lanelet_id for lanelet in lanelet_network.lanelets if not lanelet.successor}
                close_lanelets = {
                    lanelet.lanelet_id for lanelet in lanelet_network.lanelets_in_proximity(
                        self._goal_position.center, radius=1.0)}
                for lanelet_id in lanelet_assignment_shape:
                    if lanelet_id in exit_candidates and lanelet_id in close_lanelets:
                        goal_lanelet_id = lanelet_id
                        break
        else:
            goal_lanelet_id = goal_lanelet_assignment_position[0][0]
        if goal_lanelet_id is None:
            raise RespawnerSetupFailure(
                f"Failed to assign goal lanelet for scenario {str(ego_vehicle_simulation.current_scenario.scenario_id)}")
        self._goal_lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(goal_lanelet_id)

    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Respawn_Tuple:
        ego_vehicle_simulation._simulation = ego_vehicle_simulation.simulation(
            from_time_step=cast(int, self._initial_state.time_step),
            ego_vehicle=ego_vehicle_simulation.ego_vehicle,
            force=True
        )
        assert ego_vehicle_simulation.current_time_step == self._initial_state.time_step

        return self._initial_state, self._goal_position.center, self._goal_lanelet
