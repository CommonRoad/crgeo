from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple, Union, overload
from typing_extensions import Literal

import numpy as np
from commonroad.common.solution import VehicleType
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.caching.cached_property import CachedProperty, EmptyCacheException
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.types import T_CountParam
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractionParams, TrafficExtractor
from commonroad_geometric.rendering.traffic_scene_renderer import RenderParams, T_Frame, TrafficSceneRenderer
from commonroad_geometric.rendering.types import Renderable
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle, VehicleModel
from commonroad_geometric.simulation.ego_simulation.planning_problem import EgoRoute
from commonroad_geometric.simulation.ego_simulation.respawning import BaseRespawner
from commonroad_geometric.simulation.ego_simulation.respawning.base_respawner import RespawnerSetupFailure

logger = logging.getLogger(__name__)


class EgoVehicleSimulationNotYetStartedException(AttributeError):
    def __init__(self, attribute: str):
        message = f"EgoVehicleSimulation attribute '{attribute}' cannot be accessed as the simulation has not yet started."
        self.message = message
        super(EgoVehicleSimulationNotYetStartedException, self).__init__(message)


class EgoVehicleSimulationFinishedException(StopIteration):
    pass

@dataclass(frozen=True)
class EgoVehicleCollisionInfo:
    collision: bool
    ego_at_fault: Optional[bool]
    closest_obstacle_id: Optional[int]

@dataclass
class EgoVehicleSimulationOptions:
    """
        vehicle_model (VehicleModel):
        vehicle_type (VehicleType):
    """
    vehicle_model: VehicleModel = VehicleModel.KS
    vehicle_type: VehicleType = VehicleType.BMW_320i


class EgoVehicleSimulation(Renderable, AutoReprMixin):
    def __init__(
        self,
        simulation: BaseSimulation,
        control_space: BaseControlSpace,
        # ego_route: EgoRoute,
        respawner: BaseRespawner,
        traffic_extractor: Optional[TrafficExtractor] = None,
        planning_problem_set: Optional[PlanningProblemSet] = None,
        renderers: Optional[List[TrafficSceneRenderer]] = None,
        options: Optional[EgoVehicleSimulationOptions] = None
    ) -> None:
        """
        Orchestrator object which keeps track of the state of the ego vehicle of the simulation.
        For the SUMO simulation the _ego_vehicle_state is tracked with methods of the _ego_vehicle ContinuousVehicle.
        For the Extractor the _ego_vehicle is treated as an DynamicObstacle allowing access to collision information.

        The sumo_simulation_interface holds the ground truth for the current time step.
        All other time step references have to be offset correctly from this time step.

        Args:
            simulation (SumoSimulation):
            traffic_extractor (VehicleExtractor):
        """
        options = options or EgoVehicleSimulationOptions()
        self._options = options
        self._simulation = simulation
        self._control_space = control_space
        self._respawner = respawner
        self._renderers = renderers if renderers is not None else []
        self._traffic_extractor = traffic_extractor
        self._initial_planning_problem_set = planning_problem_set
        self._initial_planning_problem = next(iter(planning_problem_set.planning_problem_dict.values())) if planning_problem_set is not None and len(planning_problem_set.planning_problem_dict) > 0 else None
        self._ego_vehicle = EgoVehicle(
            vehicle_model=options.vehicle_model,
            vehicle_type=options.vehicle_type,
            dt=simulation.dt,
            ego_route=None #ego_route,
        )
        self._reset_count: int = 0
        self._respawn_count: int = 0
        self._steps_since_reset: int = 0
        self._cached_data = CachedProperty(CommonRoadData)
        self._cached_has_collision = CachedProperty(EgoVehicleCollisionInfo)
        self._cached_has_reached_goal = CachedProperty(bool)

    @property
    def simulation(self) -> BaseSimulation:
        return self._simulation

    @property
    def current_scenario(self) -> Scenario:
        return self._simulation.current_scenario

    @property
    def control_space(self) -> BaseControlSpace:
        return self._control_space

    @property
    def ego_route(self) -> EgoRoute:
        if self._ego_vehicle.ego_route is None:
            raise AttributeError("self.ego_route is None")
        return self._ego_vehicle.ego_route

    @ego_route.setter
    def ego_route(self, value: EgoRoute) -> None:
        self._ego_vehicle.ego_route = value

    @property
    def planning_problem(self) -> Optional[PlanningProblem]:
        if self._ego_vehicle.ego_route is None:
            return self._initial_planning_problem
        if self.ego_route is None:
            raise AttributeError("self.ego_route is None")
        return self.ego_route.planning_problem

    @property
    def planning_problem_set(self) -> Optional[PlanningProblemSet]:
        if self._ego_vehicle.ego_route is None:
            return self._initial_planning_problem_set
        if self.ego_route is None:
            raise AttributeError("self.ego_route is None")
        return self.ego_route.planning_problem_set

    @planning_problem_set.setter
    def planning_problem_set(self, value: PlanningProblemSet) -> None:
        if self.ego_vehicle.ego_route is None:
            self.ego_vehicle.ego_route = EgoRoute(
                simulation=self.simulation,
                planning_problem_set=value
            )
        else:
            self.ego_vehicle.ego_route.planning_problem_set = value

    @property
    def respawner(self) -> BaseRespawner:
        return self._respawner

    @property
    def traffic_extractor(self) -> Optional[TrafficExtractor]:
        return self._traffic_extractor

    @traffic_extractor.setter
    def traffic_extractor(self, value: TrafficExtractor) -> None:
        self._traffic_extractor = value

    @property
    def ego_vehicle(self) -> EgoVehicle:
        return self._ego_vehicle

    @property
    def current_non_ego_obstacles(self) -> List[DynamicObstacle]:
        all_obstacles = self._simulation.current_obstacles
        non_ego_obstacles = [obstacle for obstacle in all_obstacles if obstacle.obstacle_id != -1]
        return non_ego_obstacles

    @property
    def current_non_ego_obstacle_ids(self) -> Set[int]:
        return {obstacle.obstacle_id for obstacle in self.current_non_ego_obstacles}

    @property
    def current_num_non_ego_obstacles(self) -> int:
        return len(self.current_non_ego_obstacles)

    @property
    def initial_time_step(self) -> int:
        return self._simulation.initial_time_step

    @property
    def final_time_step(self) -> T_CountParam:
        return self._simulation.final_time_step

    @property
    def current_time_step(self) -> int:
        return self._simulation.current_time_step

    @current_time_step.setter
    def current_time_step(self, value: int) -> None:
        self._simulation.current_time_step = value

    def _assert_consistent_timesteps(self) -> None:
        assert self.current_time_step == self.ego_vehicle.state.time_step, f"Conflicting timestep between simulation and ego vehicle (sim={self.current_time_step} != ego={self.ego_vehicle.state.time_step})"

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def steps_since_reset(self) -> int:
        return self._steps_since_reset

    @property
    def respawn_count(self) -> int:
        return self._respawn_count

    @property
    def dt(self) -> float:
        return self.simulation.dt

    @property
    def current_lanelet_ids(self) -> List[int]:
        if self.ego_vehicle.ego_route is None:
            raise AttributeError("self.ego_vehicle.ego_route is None")
        if self.ego_vehicle.obstacle_id is None:
            raise AttributeError("self.ego_vehicle.obstacle_id is None")
        try:
            lanelet_assignment = self.simulation.obstacle_id_to_lanelet_id[self.ego_vehicle.obstacle_id]
        except KeyError:
            return []
        route_lanelet_ids: List[int] = []
        other_lanelet_ids: List[int] = []
        for lanelet_id in lanelet_assignment:
            if lanelet_id in self.ego_vehicle.ego_route.lanelet_id_route_set:
                route_lanelet_ids.append(lanelet_id)
            else:
                other_lanelet_ids.append(lanelet_id)
        current_lanelet_ids = route_lanelet_ids + other_lanelet_ids
        return current_lanelet_ids

    @property
    def current_lanelets(self) -> List[Lanelet]:
        current_lanelets = list(map(self._simulation.current_scenario.lanelet_network.find_lanelet_by_id, self.current_lanelet_ids))
        return current_lanelets

    @property
    def current_lanelet_center_polyline(self) -> Optional[ContinuousPolyline]:
        if len(self.current_lanelet_ids) > 0:
            return self._simulation.get_lanelet_center_polyline(self.current_lanelet_ids[0])
        return None

    def respawn(self, respawn_until_success: bool = False) -> bool:
        while True:
            try:
                self._respawner.respawn(self)
                respawn_success = True
            except RespawnerSetupFailure:
                respawn_success = False
            if not respawn_success:
                logger.warning(f"Failed to spawn ego vehicle in scenario {self.simulation.current_scenario.scenario_id}")
                if not respawn_until_success:
                    return False
            if not respawn_until_success or respawn_success:
                break
        self.control_space.reset(ego_vehicle_simulation=self)
        self._respawn_count += 1
        self.simulation.assign_unassigned_obstacles(obstacles=[self.ego_vehicle.as_dynamic_obstacle])
        self.simulation._update_obstacle_index_mappings(obstacle_id_to_lanelet_id_t={})
        logger.debug(f"Ego vehicle {self.ego_vehicle.obstacle_id} now located at lanelets {self.simulation.obstacle_id_to_lanelet_id[self.ego_vehicle.obstacle_id]} at time-step {self.current_time_step}")
        return True

    def start(self, respawn_until_success: bool = False) -> bool:
        self.simulation.start()
        self._simulation = self.simulation(
            from_time_step=0,
            ego_vehicle=self.ego_vehicle
        )
        # simulation is now running
        self._reset_cache()
        respawn_success = self.respawn(respawn_until_success=respawn_until_success)
        if not respawn_success:
            return False
        self._assert_consistent_timesteps()
        return True
        # self.extract_data()

    def close(self) -> None:
        self._reset_cache()
        self.simulation.close()

    def reset(self, respawn_until_success: bool = False) -> bool:
        self.control_space.reset(ego_vehicle_simulation=self)
        try:
            self.simulation.despawn_ego_vehicle(self.ego_vehicle)
        except KeyError:
            pass
        self.simulation.reset()
        assert self.simulation.current_time_step == self.simulation.initial_time_step
        self._reset_cache()
        respawn_success = self.respawn(respawn_until_success=respawn_until_success)
        if not respawn_success:
            return False
        self._assert_consistent_timesteps()
        # self.extract_data()
        self._reset_count += 1
        self._steps_since_reset = 0
        return True
        # further sanity checks, enable to debug
        # lanelet_id_route = self.ego_vehicle.ego_route.lanelet_id_route
        # assert len(lanelet_id_route) > 0
        # current_lanelet_candidates = [lid for lid in self.simulation.obstacle_id_to_lanelet_id[self.ego_vehicle.obstacle_id] if lid in lanelet_id_route]
        # assert len(current_lanelet_candidates) > 0
        # assert len(self.simulation.obstacle_id_to_lanelet_id[self.ego_vehicle.obstacle_id]) > 0

    def step(self, action: np.ndarray) -> Iterable[Tuple[int, Scenario]]:  # type: ignore
        self._steps_since_reset += 1
        try:
            for _ in self.control_space.step(
                ego_vehicle_simulation=self,
                action=action
            ):
                yield next(self.simulation)
        except StopIteration as e:
            return

    def extract_data(
        self,
        use_cached: bool = False
    ) -> CommonRoadData:
        if self.traffic_extractor is None:
            raise AttributeError("Cannot extract data without a provided traffic_extractor")
        cache = self._cached_data
        if cache.is_settable(self.current_time_step, overwrite=not use_cached):
            value = self.traffic_extractor.extract(
                TrafficExtractionParams(
                    index=self.current_time_step,
                    ego_vehicle=self.ego_vehicle,
                )
            )
            cache.set(time_step=self.current_time_step, value=value)
        return cache.value

    def check_if_has_collision(
        self,
        use_cached: bool = False,
        assign_blame: bool = False
    ) -> EgoVehicleCollisionInfo:
        cache = self._cached_has_collision
        write_cache = cache.is_settable(self.current_time_step, overwrite=not use_cached)
        if write_cache:
            has_collision = self.simulation.collision_checker.collide(self._ego_vehicle.collision_object)
        else:
            return cache.value

        return_struct: EgoVehicleCollisionInfo

        if has_collision and assign_blame:
            # TODO: Cleanup and move logic somewhere else
            if self.current_time_step == self.initial_time_step:
                logger.warn("Ego vehicle collided at initial time-step")
            current_lanelet_id = self.current_lanelets[0].lanelet_id
            current_lanelet_polyline = self._simulation.get_lanelet_center_polyline(current_lanelet_id)
            obstacle_ids = self._simulation.get_obstacles_on_lanelet(
                current_lanelet_id,
                ignore_ids={self.ego_vehicle.obstacle_id}
            )
            if len(obstacle_ids) == 0:
                return_struct = EgoVehicleCollisionInfo(
                    collision=True,
                    ego_at_fault=True,
                    closest_obstacle_id=None
                )
            else:
                obstacle_states = [
                    state_at_time(
                        self._simulation.current_scenario._dynamic_obstacles[oid], 
                        self.current_time_step,
                        assume_valid=True
                    ) for oid in obstacle_ids
                ]
                distances = [
                    np.linalg.norm(
                        self.ego_vehicle.state.position - obstacle_states[i].position
                    ) for i, oid in enumerate(obstacle_ids)
                ]
                closest_obstacle_idx = min(range(len(distances)), key=lambda i: distances[i])
                closest_obstacle_arclength = current_lanelet_polyline.get_projected_arclength(
                    obstacle_states[closest_obstacle_idx].position
                )
                closest_obstacle_id = obstacle_ids[closest_obstacle_idx]
                ego_arclength = current_lanelet_polyline.get_projected_arclength(
                    self.ego_vehicle.state.position
                )
                ego_at_fault = closest_obstacle_arclength > ego_arclength or \
                    abs(relative_orientation(
                        self.ego_vehicle.state.orientation,
                        obstacle_states[closest_obstacle_idx].orientation
                    )) > np.pi / 6 # TODO
                return_struct = EgoVehicleCollisionInfo(
                    collision=True,
                    ego_at_fault=ego_at_fault,
                    closest_obstacle_id=closest_obstacle_id
                )
        else:
            return_struct = EgoVehicleCollisionInfo(
                collision=has_collision,
                ego_at_fault=None,
                closest_obstacle_id=None
            )

        if write_cache:
            cache.set(time_step=self.current_time_step, value=return_struct)

        return return_struct

    def check_if_has_reached_goal(self, use_cached: bool = False) -> bool:
        cache = self._cached_has_reached_goal
        if cache.is_settable(self.current_time_step, overwrite=not use_cached):
            value = self.ego_route.check_if_has_reached_goal(ego_state=self.ego_vehicle.state)
            cache.set(time_step=self.current_time_step, value=value)
        return cache.value

    def check_if_completed_route(self) -> bool:
        polyline = self.ego_route.planning_problem_path_polyline
        if polyline is None:
            return True
        completed_route = polyline.get_projected_arclength(self.ego_vehicle.state.position, relative=True) >= 1.0
        return completed_route

    def check_if_offroad(self) -> bool:
        return len(self.current_lanelet_ids) == 0

    def check_if_offroute(self) -> bool:
        return len(self.current_lanelet_ids) == 0 or self.current_lanelet_ids[0] not in self.ego_route.lanelet_id_route_set

    def check_if_violates_friction(self) -> bool:
        return self.ego_vehicle.violate_friction

    @overload
    def render(
        self,
        renderers: None,
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[True],
        **render_kwargs: Any
    ) -> np.ndarray:
        ...  # no renderer (creating new)

    @overload
    def render(
        self,
        renderers: TrafficSceneRenderer,
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[True],
        **render_kwargs: Any
    ) -> np.ndarray:
        ...  # single renderer

    @overload
    def render(
        self,
        renderers: List[TrafficSceneRenderer],
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[True],
        **render_kwargs: Any
    ) -> List[np.ndarray]:
        ...  # multiple renderers

    @overload
    def render(
        self,
        renderers: None,
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[False],
        **render_kwargs: Any
    ) -> None:
        ...  # no renderer (creating new)

    @overload
    def render(
        self,
        renderers: TrafficSceneRenderer,
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[False],
        **render_kwargs: Any
    ) -> None:
        ...  # single renderer

    @overload
    def render(
        self,
        renderers: List[TrafficSceneRenderer],
        render_params: Optional[RenderParams],
        return_rgb_array: Literal[False],
        **render_kwargs: Any
    ) -> List[None]:
        ...  # multiple renderers

    def render(
        self,
        renderers: Union[List[TrafficSceneRenderer], TrafficSceneRenderer, None] = None,
        render_params: Optional[RenderParams] = None,
        return_rgb_array: bool = False,
        **render_kwargs: Any
    ) -> Union[T_Frame, List[T_Frame]]:
        render_params = render_params or RenderParams()

        if renderers is None:
            if not self._renderers:
                self._renderers = [TrafficSceneRenderer()]
            renderers = self._renderers

        render_kwargs = render_params.render_kwargs or {}
        if self.check_if_has_reached_goal() and 'ego_vehicle_color' not in render_kwargs:
            render_kwargs['ego_vehicle_color'] = (0.1, 0.1, 0.8, 1.0)
        if self.simulation.collision_checker.collide(self.ego_vehicle.collision_object):
            render_kwargs['ego_vehicle_color'] = (1.0, 0.0, 0.0, 1.0)

        try:
            render_params.data = self._cached_data.value
        except EmptyCacheException:
            pass
        render_params.ego_vehicle = self.ego_vehicle
        render_params.render_kwargs = render_kwargs
        render_params.ego_vehicle_simulation = self

        return self._simulation.render(
            renderers=renderers,
            render_params=render_params,
            return_rgb_array=return_rgb_array,
            **render_kwargs
        )

    def __str__(self) -> str:
        s = "EgoVehicleSimulation\n"
        s += f"- simulation_cls={self.simulation.__class__.__name__}\n"
        s += f"- ego_vehicle={str(self.ego_vehicle)}\n"
        s += f"- current_time_step={self.current_time_step}\n"
        s += f"- current_scenario={str(self.current_scenario)}"
        return s

    def _reset_cache(self) -> None:
        self._cached_data.clear()
        self._cached_has_collision.clear()
        self._cached_has_reached_goal.clear()
