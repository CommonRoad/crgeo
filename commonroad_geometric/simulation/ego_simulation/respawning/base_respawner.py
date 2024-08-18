from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import GoalRegion, PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.state import State, InitialState, KSState

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.common.io_extensions.lanelet_network import lanelet_orientation_at_position

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


logger = logging.getLogger(__name__)

T_Respawn_Tuple = Tuple[State, np.ndarray, Lanelet]


class RespawnerSetupFailure(RuntimeError):
    pass


@dataclass
class BaseRespawnerOptions:
    goal_region_length: float = 4.0
    goal_region_width: float = 2.5
    max_respawn_attempts: int = 10
    throw_on_failure: bool = True
    min_curvature: Optional[float] = None
    max_curvature: Optional[float] = None
    use_cached: bool = False


class BaseRespawner(ABC, AutoReprMixin, StringResolverMixin):
    """
    Base class for spawning the ego vehicle in the traffic environment as well as
    defining the upcoming planning problem. Typically executed on episode resets.
    """

    def __init__(
        self,
        options: BaseRespawnerOptions
    ) -> None:
        self._options = options
        self.rng = Random()
        self._respawn_cache: Dict[str, T_Respawn_Tuple] = {}

    @property
    def options(self) -> BaseRespawnerOptions:
        return self._options

    def respawn(self, ego_vehicle_simulation: EgoVehicleSimulation) -> bool:
        scenario_id = str(ego_vehicle_simulation.current_scenario.scenario_id)
        if self.options.use_cached and scenario_id in self._respawn_cache:
            respawn_tuple = self._respawn_cache[scenario_id]
            initial_state, goal_position, goal_lanelet = respawn_tuple
            ego_vehicle_simulation.planning_problem_set = BaseRespawner.create_planning_problem_set(
                ego_vehicle_simulation=ego_vehicle_simulation,
                initial_state=initial_state,
                goal_region_length=self.options.goal_region_length,
                goal_region_width=self.options.goal_region_width,
                goal_position=goal_position,
                goal_lanelet=goal_lanelet
            )
        else:
            respawn_tuple = self._prepare_respawn_tuple(ego_vehicle_simulation)
            initial_state, goal_position, goal_lanelet = respawn_tuple
            if self.options.use_cached:
                self._respawn_cache[scenario_id] = respawn_tuple

        self._activate_state(
            ego_vehicle_simulation=ego_vehicle_simulation,
            initial_state=initial_state
        )
        logger.debug(f"{type(self).__name__} respawned ego vehicle to initial state")

        return True

    def _prepare_respawn_tuple(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation
    ) -> T_Respawn_Tuple:

        self._setup(ego_vehicle_simulation)

        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        ego_vehicle_simulation.simulation.despawn_ego_vehicle(ego_vehicle)
        success_spawn = False

        def validate_spawn_location() -> bool:
            collision_struct = ego_vehicle_simulation.check_if_has_collision(use_cached=False, assign_blame=False)
            if collision_struct.collision:
                return False
            trajectory = ego_vehicle_simulation.ego_route.planning_problem_path_polyline

            assert trajectory is not None

            if self.options.min_curvature is not None or self.options.max_curvature is not None:
                curvature_arr = np.abs(trajectory.get_curvature_arr())
                max_curvature = curvature_arr.max()
                if self.options.max_curvature is not None:
                    if max_curvature > self.options.max_curvature:
                        return False
                if self.options.min_curvature is not None:
                    if max_curvature < self.options.min_curvature:
                        return False
            return True

        for _ in range(self._options.max_respawn_attempts):
            try:
                # Note: Modifying initial_state is dangerous since it is created elsewhere, could be the same object or different for each respawn attempt
                respawn_tuple = self._get_respawn_tuple(ego_vehicle_simulation)
                initial_state, goal_position, goal_lanelet = respawn_tuple
                ego_vehicle_simulation.planning_problem_set = BaseRespawner.create_planning_problem_set(
                    ego_vehicle_simulation=ego_vehicle_simulation,
                    initial_state=initial_state,
                    goal_region_length=self.options.goal_region_length,
                    goal_region_width=self.options.goal_region_width,
                    goal_position=goal_position,
                    goal_lanelet=goal_lanelet
                )
                assert ego_vehicle_simulation.planning_problem is not None
                ego_state = ego_vehicle_simulation.planning_problem.initial_state
                ego_state.time_step = ego_vehicle_simulation.current_time_step
                if not hasattr(ego_state, 'steering_angle'):
                    ego_state.steering_angle = 0.0
                
                ego_vehicle.reset(initial_state=ego_state)
                if not validate_spawn_location():
                    continue
                success_spawn = True
                break
            except RespawnerSetupFailure as e:
                # TODO
                # logger.error(e, exc_info=True)
                continue

        if not success_spawn and self._options.throw_on_failure:
            raise RespawnerSetupFailure()

        return respawn_tuple

    def _activate_state(self, ego_vehicle_simulation: EgoVehicleSimulation, initial_state: State) -> None:
        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        ego_vehicle.reset(initial_state=initial_state)
        ego_vehicle_simulation.simulation.spawn_ego_vehicle(ego_vehicle)
        if ego_vehicle_simulation.traffic_extractor is not None:
            ego_vehicle_simulation.traffic_extractor.reset_feature_computers()
        if not ego_vehicle_simulation.current_time_step == ego_vehicle.current_time_step:
            next_ego_state = deepcopy(ego_vehicle.state)
            next_ego_state.time_step = ego_vehicle_simulation.current_time_step
            ego_vehicle.set_next_state(next_ego_state)

    @abstractmethod
    def _get_respawn_tuple(self, ego_vehicle_simulation: EgoVehicleSimulation) -> T_Respawn_Tuple:
        """
        Called by respawn to determine next respawn position.

        Args:
            ego_vehicle_simulation (EgoVehicleSimulation): the simulation for which the ego vehicle has to be respawned

        |  Returns:
        |  Tuple consisting of respawn location given by:
        |       * Initial state of ego vehicle
        |       * Goal position of ego vehicle
        |       * Goal lanelet of ego vehicle
        |       * Flag: True if the (non-ego) simulation should run a step on respawn failure before the next respawn attempt.

        """
        ...

    def _setup(self, ego_vehicle_simulation: EgoVehicleSimulation) -> None:
        pass

    @staticmethod
    def create_planning_problem_set(
        ego_vehicle_simulation: EgoVehicleSimulation,
        initial_state: State,
        goal_region_length: float,
        goal_region_width: float,
        goal_position: np.ndarray,
        goal_lanelet: Lanelet
    ) -> PlanningProblemSet:
        goal_region = BaseRespawner.create_goal_region(
            ego_vehicle_simulation=ego_vehicle_simulation,
            position=goal_position,
            lanelet=goal_lanelet,
            length=goal_region_length,
            width=goal_region_width,
        )
        planning_problem_set = PlanningProblemSet([
            PlanningProblem(
                planning_problem_id=0,
                initial_state=initial_state,
                goal_region=goal_region
            )
        ])
        return planning_problem_set

    @staticmethod
    def create_goal_region(
        ego_vehicle_simulation: EgoVehicleSimulation,
        position: np.ndarray,
        length: float,
        width: float,
        lanelet: Optional[Lanelet] = None,
        recompute_position: bool = True
    ) -> GoalRegion:

        if lanelet is None:
            lanelet_id = ego_vehicle_simulation.current_scenario.lanelet_network.find_lanelet_by_position([position])[0][0]
            lanelet = ego_vehicle_simulation.simulation.find_lanelet_by_id(lanelet_id)
        if recompute_position:
            position = ego_vehicle_simulation.simulation.get_lanelet_center_polyline(
                lanelet.lanelet_id,
            ).get_projected_position(position, linear_projection=True)

        orientation = lanelet_orientation_at_position(lanelet, position)
        goal_region = GoalRegion(
            state_list=[
                InitialState(
                    time_step=Interval(0.0, 1e6),
                    position=Rectangle(
                        length=length,
                        width=width,
                        center=position,
                        orientation=orientation
                    ),
                    orientation=AngleInterval(-np.pi / 2, np.pi / 2),
                    velocity=Interval(0.0, 1e6)
                )
            ],
            lanelets_of_goal_position={
                0: [int(lanelet.lanelet_id)]
            }
        )
        return goal_region
