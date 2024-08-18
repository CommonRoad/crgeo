from __future__ import annotations

import itertools
import logging
import sys
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union, overload

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad_dc.pycrcc import CollisionChecker  # noqa
from statemachine import State, StateMachine

from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin
from commonroad_geometric.common.geometry.continuous_polyline import ContinuousPolyline
from commonroad_geometric.common.geometry.helpers import relative_orientation
from commonroad_geometric.common.io_extensions.obstacle import get_obstacle_lanelet_assignment, state_at_time
from commonroad_geometric.common.io_extensions.scenario import LANELET_ASSIGNMENT_STRATEGIES_CENTER, \
    LANELET_ASSIGNMENT_STRATEGIES_SHAPE, LaneletAssignmentStrategy, backup_scenario, \
    get_dynamic_obstacles_at_timestep, iter_dynamic_obstacles_at_timestep, iter_unassigned_dynamic_obstacles_at_timestep
from commonroad_geometric.common.types import T_CountParam, Unlimited
from commonroad_geometric.dataset.extraction.road_network.types import GraphConversionStep
from commonroad_geometric.rendering.plugins.base_renderer_plugin import T_RendererPlugin
from commonroad_geometric.rendering.traffic_scene_renderer import T_Frame, TrafficSceneRenderer, GLViewerOptions
from commonroad_geometric.rendering.types import RenderParams, Renderable
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.simulation.exceptions import SimulationNotYetStartedException, SimulationRuntimeError
from commonroad_geometric.simulation.lanelet_network_info import LaneletNetworkInfo

T_SimulationOptions = TypeVar('T_SimulationOptions', bound='BaseSimulationOptions')

logger = logging.getLogger(__name__)


@dataclass
class BaseSimulationOptions:
    """
        dt (float): Positive time delta between each time-step. Integral granularity, numerical step-size of simulation.
        step_renderers (Optional[TrafficSceneRenderer]): Needs to be present if rendering should be done on every step.
        render_plugins (Optional[List[T_RendererPlugin]]): Optional render plugins for TrafficSceneRenderer.
        render_kwargs (Dict[str, Any]): Optional kwargs for TrafficSceneRenderer.
        backup_initial_scenario (bool): True if scenarios should be deep-copied during initialization. Defaults to False.
    """
    dt: float = 0.04
    step_renderers: List[TrafficSceneRenderer] = field(default_factory=list)
    render_plugins: List[T_RendererPlugin] = field(default_factory=list)
    render_kwargs: Dict[str, Any] = field(default_factory=dict)
    backup_initial_scenario: bool = False
    backup_current_scenario: bool = False
    collision_checking: bool = True
    lanelet_assignment_order: LaneletAssignmentStrategy = LaneletAssignmentStrategy.ONLY_SHAPE
    ignore_assignment_opposite_direction: bool = True
    lanelet_graph_conversion_steps: Optional[List[GraphConversionStep]] = None
    linear_lanelet_projection: bool = False
    sort_lanelet_assignments: bool = True


class SimulationLifecycle(StateMachine):  # type: ignore
    after_init = State('Ready', initial=True)
    started = State('Started')
    running = State('Running')
    running_ego = State('Running with live ego vehicle')
    finished = State('Finished')
    closed = State('Closed')
    crashed = State('Crashed')

    start = after_init.to(started) | closed.to(started)
    run = started.to(running) | running_ego.to(running)
    run_ego = started.to(running_ego) | running.to(running_ego)
    finish = running.to(finished) | running_ego.to(finished)
    close = after_init.to(closed) | started.to(closed) | running.to(closed) | running_ego.to(closed) | finished.to(
        closed) | closed.to.itself()
    crash = after_init.to(crashed) | started.to(crashed) | running.to(crashed) | running_ego.to(crashed) | finished.to(
        crashed) | closed.to(crashed) | crashed.to.itself()
    reset = running.to(started) | running_ego.to(started) | finished.to(started) | closed.to(
        started) | started.to.itself()


class BaseSimulation(LaneletNetworkInfo, Renderable, Iterator[Tuple[int, Scenario]], Generic[T_SimulationOptions], StringResolverMixin):
    def __init__(
        self,
        initial_scenario: Union[Scenario, Path],
        options: T_SimulationOptions
    ) -> None:
        """
        Base class for simulations based on CommonRoad scenarios. Uses TrafficSceneRenderer for rendering.

        Args:
            initial_scenario (Union[Scenario, str]): Initial scenario for this simulation.
            options (T_BaseSimulationOptions): Options for this simulation.
        """
        if isinstance(initial_scenario, Path):
            initial_scenario, planning_problem_set = CommonRoadFileReader(filename=str(initial_scenario)).open()
        self._options: T_SimulationOptions = options
        initial_scenario.dt = self._options.dt

        # sorting lanelets
        initial_scenario.lanelet_network._lanelets = dict(sorted(initial_scenario.lanelet_network._lanelets.items()))

        self._initial_scenario: Scenario = backup_scenario(
            initial_scenario) if options.backup_initial_scenario else initial_scenario
        self._current_scenario: Scenario = backup_scenario(
            initial_scenario) if options.backup_current_scenario else initial_scenario
        self._initial_time_step, self._final_time_step = self._get_time_step_bounds()

        self._lifecycle: SimulationLifecycle = SimulationLifecycle()

        self._current_time_step: Optional[int] = None
        self._current_collision_checker: Optional[CollisionChecker] = None
        self._current_run: Optional[Iterator[int]] = None
        self._current_run_start: Optional[int] = None
        self._current_ego_vehicle: Optional[EgoVehicle] = None

        self._time_step_to_obstacles: Dict[int, List[DynamicObstacle]] = defaultdict(list)

        self._obstacle_id_to_obstacle_idx_t: Dict[int, int] = {}
        self._obstacle_idx_to_obstacle_id_t: Dict[int, int] = {}
        # Obstacle id remains fixed and is unique, index can change between time-step and next time-step
        self._obstacle_id_to_lanelet_id_t: Dict[int, List[int]] = {}
        self._obstacle_id_to_lanelet_id_t_minus_one: Dict[int, List[int]] = {}
        self._step_rendering_disabled: bool = False

        super().__init__(
            scenario=self.current_scenario,
            graph_conversion_steps=self._options.lanelet_graph_conversion_steps
        )

    @property
    def options(self) -> BaseSimulationOptions:
        return self._options

    @property
    def initial_scenario(self) -> Scenario:
        return self._initial_scenario

    @property
    def current_scenario(self) -> Scenario:
        return self._current_scenario

    @current_scenario.setter
    def current_scenario(self, value: Scenario) -> None:
        raise ValueError(f"Current scenario is immutable for {type(self).__name__}! "
                         f"Full simulation needs to be available in initial_scenario.")

    @property
    def dt(self) -> float:
        return self._options.dt

    @property
    def lifecycle(self) -> SimulationLifecycle:
        return self._lifecycle

    @property
    @abstractmethod
    def collision_checker(self) -> CollisionChecker:
        ...

    @property
    def current_obstacles(self) -> List[DynamicObstacle]:
        if self.current_time_step not in self._time_step_to_obstacles:
            self._time_step_to_obstacles[self.current_time_step] = get_dynamic_obstacles_at_timestep(
                self.current_scenario,
                self.current_time_step
            )
        return self._time_step_to_obstacles[self.current_time_step]

    def get_current_obstacle_state(self, obstacle: DynamicObstacle) -> State:
        return state_at_time(
            obstacle=obstacle,
            time_step=self.current_time_step,
            assume_valid=True
        )

    @property
    def current_obstacle_ids(self) -> Set[int]:
        return {obstacle.obstacle_id for obstacle in self.current_obstacles}

    @property
    def current_num_obstacles(self) -> int:
        return len(self.current_obstacles)

    @property
    def current_time_step(self) -> int:
        """
        Indicates the current simulation step, i.e. the one that has been last executed

        Returns:
            Current time step of simulation. Throws an AttributeError if simulation hasn't been initialized/performed a step yet.
        """
        if self._current_time_step is None:
            self.lifecycle.crash()
            raise SimulationNotYetStartedException('current_time_step')
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, time_step_to_jump_to: int) -> None:
        """
        Allows wild jumps through scenario within bounds of initial_time_step and final_time_step,
        as full simulation run data is already available.

        Args:
            time_step_to_jump_to (int): the time-step that should be jumped to,

        Returns:
            None, raise ValueError if out of bounds for scenario

        """
        if self.final_time_step is not Unlimited:
            if self.initial_time_step <= time_step_to_jump_to <= self.final_time_step:
                self._current_time_step = time_step_to_jump_to
            else:
                raise ValueError(f"Time-step: {time_step_to_jump_to} is not within bounds [{self.initial_time_step}, "
                                 f"{self.final_time_step}] of {type(self).__name__}.")
        else:
            if self.initial_time_step <= time_step_to_jump_to:
                self._current_time_step = time_step_to_jump_to
            else:
                raise ValueError(f"Time-step: {time_step_to_jump_to} is not within bounds "
                                 f"[{self.initial_time_step}, inf] of {type(self).__name__}.")

    @abstractmethod
    def _get_time_step_bounds(self) -> Tuple[int, T_CountParam]:
        ...

    @property
    def initial_time_step(self) -> int:
        return self._initial_time_step

    @property
    def final_time_step(self) -> T_CountParam:
        return self._final_time_step

    @property
    def current_run_start(self) -> Optional[int]:
        """
        Indicates the starting time step of the current simulation run, i.e. the one that has been executed first

        Returns:
            Starting time step of current simulation run. Throws an AttributeError if simulation hasn't been initialized/performed a step yet.
        """
        if self._current_run_start is None:
            self.lifecycle.crash()
            raise SimulationNotYetStartedException('current_run_start')
        return self._current_run_start

    @property
    def num_time_steps(self) -> T_CountParam:
        if self.final_time_step is Unlimited:
            return Unlimited
        return self.final_time_step - self.initial_time_step + 1

    @property
    def iter_time_steps(self) -> Iterable[int]:
        if self.final_time_step is Unlimited:
            return itertools.count(self.initial_time_step)
        return range(self.initial_time_step, self.final_time_step + 1)

    def __iter__(self) -> Iterator[Tuple[int, Scenario]]:
        """
        Returns self as an Iterator after configuration.

        Use like this:

           for time_step, scenario in simulation:
               # do something here

        Returns:
            Iterator ranging from self.initial_time_step to self.final_time_step, with  __next__ not using any ego_vehicle reference.
        """
        if self.lifecycle.is_running:
            # If running just return current iteration without restarting (restart can be done with reset)
            return self
        if not self.lifecycle.is_started:
            self.lifecycle.crash()
            raise SimulationNotYetStartedException("__iter__, not self.lifecycle.is_started")
        if self.final_time_step is not Unlimited and self.initial_time_step > self.final_time_step:
            raise SimulationRuntimeError(f"BaseSimulation initial time-step {self.initial_time_step} would be after "
                                         f"final time-step {self.final_time_step}")
        self._current_run = iter(self.iter_time_steps)
        self._current_run_start = self.initial_time_step
        self._current_time_step = self.initial_time_step
        self._current_ego_vehicle = None
        self.lifecycle.run()
        return self

    @overload
    def __call__(
        self,
        *,
        from_time_step: Optional[int] = None,
        to_time_step: Optional[T_CountParam] = None,
        ego_vehicle: Optional[EgoVehicle] = None,
    ) -> BaseSimulation:
        ...

    @overload
    def __call__(
        self,
        *,
        num_time_steps: Optional[T_CountParam] = None,
        ego_vehicle: Optional[EgoVehicle] = None,
    ) -> BaseSimulation:
        ...

    def __call__(
        self,
        *,
        from_time_step: Optional[int] = None,
        to_time_step: Optional[T_CountParam] = None,
        num_time_steps: Optional[T_CountParam] = None,
        ego_vehicle: Optional[EgoVehicle] = None,
        force: bool = False
    ) -> BaseSimulation:
        """
        Returns self as an Iterator after configuration.
        Important note: __iter__ is still called immediately after __call__

        Use like this:

           for time_step, scenario in simulation(from_time_step=x, to_time_step=y, ego_vehicle=ego_vehicle):
               # do something here

        Args:
            from_time_step (int, optional): Time-step from which this simulation run will start.
                Defaults to initial_time_step. Mutually exclusive with num_time_steps.
            to_time_step (int or Unlimited, optional): Time-step at which this simulation run will end.
                Defaults to final_time_step. Mutually exclusive with num_time_steps.
            num_time_steps (int or Unlimited, optional): Number of time-steps to run this simulation for, starting from
                initial_time_step. Mutually exclusive with from_time_step and to_time_step.
            ego_vehicle (EgoVehicle, optional): Ego vehicle instance. Defaults to None.

        Returns:
            Iterator ranging from from_time_step to to_time_step, with  __next__ using the passed ego_vehicle reference.
        """
        assert num_time_steps is None or from_time_step is None and to_time_step is None, \
            "You can either specify from_time_step and to_time_step or num_time_steps, they are mutually exclusive"
        if not force:
            if self.lifecycle.is_running or self.lifecycle.is_running_ego:
                # If running just return current iteration without restarting (restart can be done with reset)
                return self
            if not self.lifecycle.is_started:
                self.lifecycle.crash()
                raise SimulationNotYetStartedException("__call__, not self.lifecycle.is_started")

        if not self.lifecycle.is_running:
            self.lifecycle.run()

        start: int = self.initial_time_step if from_time_step is None else from_time_step
        end: T_CountParam
        if num_time_steps is not None:
            end = Unlimited if num_time_steps is Unlimited else num_time_steps + start
        else:
            end = self.final_time_step if to_time_step is None else to_time_step
        if end is not Unlimited and start > end:
            raise SimulationRuntimeError(f"BaseSimulation start time-step {start} would be after end time-step {end}")

        self._current_run = iter(itertools.count(start)) if end is Unlimited else iter(range(start, end))
        # Set first time step of this run
        self._current_time_step = start
        self._current_run_start = start
        # Start with empty current obstacle_id_to_lanelet_id for new run,
        # s.t. obstacle_id_to_lanelet_id from last run is not copied into obstacle_id_to_lanelet_id_last
        self._update_obstacle_index_mappings(obstacle_id_to_lanelet_id_t={})
        # DO not switch lifecycle to running with ego here, only after spawning
        self._current_ego_vehicle = ego_vehicle
        return self

    def __next__(self) -> Tuple[int, Scenario]:
        """
        Advances the simulation to the next time step.

        Returns:
            Raises StopIteration if the simulation is finished (i.e. has reached the final time step/to_time_step).
        """
        if not self.lifecycle.running:
            raise SimulationRuntimeError(f"Invalid call to __next__ in BaseSimulation lifecycle state {self.lifecycle}")
        if self._current_run is None:
            logger.debug(f"self._current_run is None, using simulation bounds self.iter_time_steps as fallback")
            self._current_run = iter(self.iter_time_steps)
        try:
            self._current_time_step = next(self._current_run)
        except StopIteration:
            self.lifecycle.finish()
            raise StopIteration

        # Calling internal step function
        # try:
        self._step(ego_vehicle=self._current_ego_vehicle if self.lifecycle.is_running_ego else None)
        # except Exception as e:
        #     logger.error(e, exc_info=True)
        #     raise SimulationRuntimeError() from e
        # self.current_obstacles needs to be up-to-date before this is called
        self._update_obstacle_index_mappings(obstacle_id_to_lanelet_id_t=self._obstacle_id_to_lanelet_id_t)
        # Now lanelet information and indices for self.current_time_step should be available for each self.current_obstacles

        if self._options.step_renderers and not self._step_rendering_disabled:
            self.render(
                renderers=self._options.step_renderers,
                render_params=RenderParams(
                    ego_vehicle=self._current_ego_vehicle if self.lifecycle.is_running_ego else None,
                    simulation=self
                ),
                **self._options.render_kwargs or {}
            )

        return self.current_time_step, self.current_scenario

    def disable_step_rendering(self) -> None:
        self._step_rendering_disabled = True

    def activate_step_rendering(self) -> None:
        self._step_rendering_disabled = True

    def spawn_ego_vehicle(self, ego_vehicle: EgoVehicle) -> None:
        assert ego_vehicle.as_dynamic_obstacle is not None
        if self.current_scenario._is_object_id_used(ego_vehicle.as_dynamic_obstacle.obstacle_id):
            self.current_scenario.remove_obstacle(ego_vehicle.as_dynamic_obstacle)
        spawn_time_step = ego_vehicle.state.time_step
        self.current_scenario.add_objects(ego_vehicle.as_dynamic_obstacle)
        self.assign_unassigned_obstacles(
            obstacles=[ego_vehicle.as_dynamic_obstacle]
        )
        self._time_step_to_obstacles[spawn_time_step].append(ego_vehicle.as_dynamic_obstacle)
        self._append_obstacle_index_mapping(obstacle=ego_vehicle.as_dynamic_obstacle)
        # Internal implementation has to change lifecycle to running with ego vehicle
        self._spawn_ego_vehicle(ego_vehicle)

    def despawn_ego_vehicle(self, ego_vehicle: Optional[EgoVehicle] = None) -> None:
        # If running with ego vehicle, switch to running without ego vehicle
        if self.lifecycle.is_running_ego:
            self.lifecycle.run()
        if ego_vehicle is None or ego_vehicle.as_dynamic_obstacle is None:
            ego_obstacle_id_by_convention = -1
            if self.current_scenario._is_object_id_used(object_id=ego_obstacle_id_by_convention):
                ego_obstacle = self.current_scenario._dynamic_obstacles.get(ego_obstacle_id_by_convention, None)
                ego_obstacle.prediction.shape_lanelet_assignment = None
                ego_obstacle.prediction.center_lanelet_assignment = None
                self.current_scenario.remove_obstacle(obstacle=ego_obstacle)
            return
        if self.current_scenario._is_object_id_used(ego_vehicle.as_dynamic_obstacle.obstacle_id):
            try:
                # self.current_scenario.remove_obstacle(ego_vehicle.as_dynamic_obstacle)
                del self.current_scenario._dynamic_obstacles[ego_vehicle.as_dynamic_obstacle.obstacle_id]
                self.current_scenario._id_set.remove(ego_vehicle.as_dynamic_obstacle.obstacle_id)
            except KeyError:
                logger.exception(f"Failed to remove ego vehicle obstacle", stack_info=True)
            # Might need to implement _remove_obstacle_index_mapping method here

    def start(self) -> None:
        self.lifecycle.start()
        self._current_time_step = self.initial_time_step
        self._start()
        self._update_obstacle_index_mappings(obstacle_id_to_lanelet_id_t={})

    def close(self) -> None:
        # self._current_collision_checker = None
        self._current_run = None
        self._current_ego_vehicle = None

        self._obstacle_id_to_obstacle_idx_t = {}
        self._obstacle_idx_to_obstacle_id_t = {}

        self._obstacle_id_to_lanelet_id_t = {}
        self._obstacle_id_to_lanelet_id_t_minus_one = {}

        self._close()
        try:
            if not self.lifecycle.is_crashed:
                self.lifecycle.close()
        except AttributeError:
            pass

    def __enter__(self) -> BaseSimulation:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def reset(self) -> None:
        self.lifecycle.reset()

        self._current_time_step = self.initial_time_step
        self._current_run = None
        self._current_ego_vehicle = None

        self._obstacle_id_to_obstacle_idx_t = {}
        self._obstacle_idx_to_obstacle_id_t = {}

        self._obstacle_id_to_lanelet_id_t = {}
        self._obstacle_id_to_lanelet_id_t_minus_one = {}

        # Reset of self._time_step_to_obstacles must be handled by implementation not BaseSimulation!
        self._reset()
        self._update_obstacle_index_mappings(obstacle_id_to_lanelet_id_t=self._obstacle_id_to_lanelet_id_t)

    @abstractmethod
    def _step(self, ego_vehicle: Optional[EgoVehicle] = None) -> None:
        pass

    @abstractmethod
    def _spawn_ego_vehicle(self, ego_vehicle: EgoVehicle) -> None:
        pass

    @abstractmethod
    def _start(self) -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        pass

    @abstractmethod
    def _reset(self) -> None:
        pass

    @property
    def obstacle_id_to_obstacle_idx(self) -> Dict[int, int]:
        return self._obstacle_id_to_obstacle_idx_t

    @property
    def obstacle_idx_to_obstacle_id(self) -> Dict[int, int]:
        return self._obstacle_idx_to_obstacle_id_t

    @property
    def obstacle_id_to_lanelet_id(self) -> Dict[int, List[int]]:
        return self._obstacle_id_to_lanelet_id_t

    @property
    def obstacle_id_to_lanelet_id_last(self) -> Dict[int, List[int]]:
        return self._obstacle_id_to_lanelet_id_t_minus_one

    def get_obstacles_on_lanelet(
        self,
        lanelet_id: int,
        ignore_ids: Optional[Set[int]] = None
    ) -> List[int]:
        # TODO: optimize
        lanelet_id = int(lanelet_id)
        obstacles: List[int] = []
        for obstacle_id, lanelet_ids in self.obstacle_id_to_lanelet_id.items():
            if ignore_ids is not None and obstacle_id in ignore_ids:
                continue
            for _lanelet_id in lanelet_ids:
                if _lanelet_id == lanelet_id:
                    obstacles.append(obstacle_id)
                    break
        return obstacles

    def has_lanelet_assignment(self, obstacle_id: int) -> bool:
        return len(self.obstacle_id_to_lanelet_id[obstacle_id]) > 0

    @property
    def renderers(self) -> List[TrafficSceneRenderer]:
        return self._options.step_renderers

    def _update_obstacle_index_mappings(
        self,
        obstacle_id_to_lanelet_id_t: Dict[int, List[int]]
    ) -> None:
        """
        Update obstacle_id <-> obstacle_idx and obstacle_id -> lanelet mappings.
        obstacle_id is unique and does not change
        obstacle_idx depends on iteration order and can thus change in between time-steps, this has to be accounted for.

        Args:
            obstacle_id_to_lanelet_id_t (Dict[int, Optional[int]]): self.obstacle_id_to_lanelet_id_t or {}

        Returns:
            None
        """
        # Assign dynamic obstacles objects to lanelets before running update
        self.assign_unassigned_obstacles()

        # t_plus_one is t after this function is finished
        # t is t_minus_one after this function is finished
        obstacle_id_to_obstacle_idx_t_plus_one: Dict[int, int] = {}
        obstacle_idx_to_obstacle_id_t_plus_one: Dict[int, int] = {}
        obstacle_id_to_lanelet_id_t_plus_one: Dict[int, List[int]] = {}

        for obstacle_idx_t_plus_one, obstacle in enumerate(iter_dynamic_obstacles_at_timestep(self.current_scenario, self.current_time_step)):
            obstacle_id = obstacle.obstacle_id

            # Update previous mapping in loop, instead of just copying as some obstacles from now t_minus_one might not be current anymore
            self._obstacle_id_to_lanelet_id_t_minus_one[obstacle_id] = obstacle_id_to_lanelet_id_t.get(obstacle_id, [])

            # Create new obstacle_id <-> obstacle_idx_t_plus_one mappings as indices might have changed with addition of new vehicles
            obstacle_id_to_obstacle_idx_t_plus_one[obstacle_id] = obstacle_idx_t_plus_one
            obstacle_idx_to_obstacle_id_t_plus_one[obstacle_idx_t_plus_one] = obstacle_id

            # Assign lanelet for t_plus_one
            lanelet_assignment: List[int] = get_obstacle_lanelet_assignment(
                obstacle=obstacle,
                time_step=self.current_time_step,
                use_center_lanelet_assignment=self._options.lanelet_assignment_order in LANELET_ASSIGNMENT_STRATEGIES_CENTER,
                use_shape_lanelet_assignment=self._options.lanelet_assignment_order in LANELET_ASSIGNMENT_STRATEGIES_SHAPE,
            )

            assert isinstance(lanelet_assignment, list)
            # if not lanelet_assignment:
            #     warnings.warn(f"Obstacle {obstacle.obstacle_id} missing lanelet assignment at time-step {self.current_time_step}.")

            obstacle_id_to_lanelet_id_t_plus_one[obstacle_id] = lanelet_assignment

        self._obstacle_id_to_obstacle_idx_t = obstacle_id_to_obstacle_idx_t_plus_one
        self._obstacle_idx_to_obstacle_id_t = obstacle_idx_to_obstacle_id_t_plus_one
        self._obstacle_id_to_lanelet_id_t = obstacle_id_to_lanelet_id_t_plus_one

    def _append_obstacle_index_mapping(self, obstacle: DynamicObstacle) -> None:
        """
        Appends a single obstacle to the current obstacle_id <-> obstacle_idx and obstacle_id -> lanelet mappings.

        Args:
            obstacle (DynamicObstacle): the obstacle which should be appended

        Returns:
            None
        """
        if self._obstacle_idx_to_obstacle_id_t:
            next_available_idx = max(self._obstacle_idx_to_obstacle_id_t) + 1
        else:
            next_available_idx = 0
        obstacle_id = obstacle.obstacle_id
        self._obstacle_id_to_obstacle_idx_t[obstacle_id] = next_available_idx
        self._obstacle_idx_to_obstacle_id_t[next_available_idx] = obstacle_id
        # Assign t_minus one before updating for t
        self._obstacle_id_to_lanelet_id_t_minus_one[obstacle_id] = self._obstacle_id_to_lanelet_id_t.get(obstacle_id,
                                                                                                         [])
        self._obstacle_id_to_lanelet_id_t[obstacle_id] = get_obstacle_lanelet_assignment(obstacle,
                                                                                         self.current_time_step)

    def assign_unassigned_obstacles(
        self,
        obstacles: Optional[List[DynamicObstacle]] = None
    ) -> None:

        scenario = self.current_scenario
        time_step = self.current_time_step
        assignment_order = self._options.lanelet_assignment_order
        obstacles = self.current_obstacles if obstacles is None else obstacles

        unassigned_obstacles = iter_unassigned_dynamic_obstacles_at_timestep(
            scenario=scenario,
            time_step=time_step,
            obstacles=obstacles,
            check_center_lanelet_assignment=assignment_order in LANELET_ASSIGNMENT_STRATEGIES_CENTER,
            check_shape_lanelet_assignment=assignment_order in LANELET_ASSIGNMENT_STRATEGIES_SHAPE,
        )
        for obstacle in unassigned_obstacles:
            obstacle_state = state_at_time(obstacle, time_step, assume_valid=True)

            center_assignment_success = False
            lanelet_assignment = []

            if assignment_order in {LaneletAssignmentStrategy.ONLY_CENTER,
                                    LaneletAssignmentStrategy.CENTER_FALLBACK_SHAPE}:
                lanelet_assignment = scenario.lanelet_network.find_lanelet_by_position(
                    point_list=[obstacle_state.position]
                )[0]
                center_assignment_success = len(lanelet_assignment) > 0
            if not center_assignment_success and assignment_order in LANELET_ASSIGNMENT_STRATEGIES_SHAPE:
                # Fallback if assignment by position fails,
                # happens in edge case where obstacle jitters on border of two lanelets during lane change
                obstacle_shape = obstacle.obstacle_shape.rotate_translate_local(obstacle_state.position,
                                                                                angle=obstacle_state.orientation)
                lanelet_assignment = scenario.lanelet_network.find_lanelet_by_shape(obstacle_shape)

            if self.options.sort_lanelet_assignments:
                lanelet_polylines = list(map(self.get_lanelet_center_polyline, lanelet_assignment))
                lanelet_assignment = [lanelet_assignment[i] for i in self._sorted_lanelet_indices(
                    lanelet_polylines,
                    orientation=obstacle_state.orientation,
                    position=obstacle_state.position
                )]
                lanelet_assignment_keep = []

                # removing assignment to adjacent opposite lanelets
                for l_outer_idx in range(len(lanelet_assignment)):
                    skip_assignment = False
                    l_outer = scenario.lanelet_network.find_lanelet_by_id(lanelet_assignment[l_outer_idx])
                    for l_former_idx in range(0, l_outer_idx):
                        l_former_id = lanelet_assignment[l_former_idx]
                        if l_outer.adj_left == l_former_id and not l_outer.adj_left_same_direction or \
                            l_outer.adj_right == l_former_id and not l_outer.adj_right_same_direction:
                            skip_assignment = True
                            break
                    if not skip_assignment:
                        lanelet_assignment_keep.append(lanelet_assignment[l_outer_idx])
                lanelet_assignment = lanelet_assignment_keep

            assert isinstance(lanelet_assignment, list)

            if center_assignment_success:
                if obstacle.prediction.center_lanelet_assignment is None:
                    obstacle.prediction.center_lanelet_assignment = {}
                obstacle.prediction.center_lanelet_assignment[time_step] = lanelet_assignment  # type: ignore
            else:
                if obstacle.prediction.shape_lanelet_assignment is None:
                    obstacle.prediction.shape_lanelet_assignment = {}
                obstacle.prediction.shape_lanelet_assignment[time_step] = lanelet_assignment  # type: ignore

    def get_obstacle_lanelet(self, obstacle: DynamicObstacle) -> Optional[Lanelet]:
        lanelet_assignment_ids = self.obstacle_id_to_lanelet_id.get(obstacle.obstacle_id, None)
        if lanelet_assignment_ids:
            return self.current_scenario.lanelet_network.find_lanelet_by_id(lanelet_assignment_ids[0])
        return None

    def has_changed_lanelet(self, obstacle: DynamicObstacle) -> bool:
        obstacle_id = obstacle.obstacle_id
        if obstacle_id not in self._obstacle_id_to_lanelet_id_t_minus_one or \
            self._obstacle_id_to_lanelet_id_t_minus_one.get(obstacle_id, None) is None:
            return False

        lanelet_id_t_minus_one = self._obstacle_id_to_lanelet_id_t_minus_one[obstacle_id]
        lanelet_id_t = self._obstacle_id_to_lanelet_id_t[obstacle_id]
        has_changed_lanelet = lanelet_id_t_minus_one != lanelet_id_t
        return has_changed_lanelet

    def render(
        self,
        *,
        renderers: Optional[Sequence[TrafficSceneRenderer]] = None,
        render_params: Optional[RenderParams] = None,
        return_frames: bool = False,
        **render_kwargs: Dict[str, Any]
    ) -> Sequence[T_Frame]:
        """
        Renders the current state of the simulation for each of the renderers.
        If no or partial render_params are passed, only simulation specific attributes (time_step and current_scenario)
        are rendered.

        Args:
            renderers (Sequence[TrafficSceneRenderer]): Sequence of TrafficSceneRenderer's which are called
            render_params (Optional[RenderParams]): Optional parameters which should be rendered. Defaults to None.
            return_frames (bool): Whether to return frames from the renderers. Defaults to False.
            **render_kwargs (Dict[str, Any]): Additional kwargs which will be passed to the renderers

        Returns:
            - a sequences of frames with one frame per renderer (list[np.ndarray]) if return_frames is True
            - an empty list ([]) if return_frames is False
        """
        if renderers is None:
            renderers = []
        if not renderers and not self.options.step_renderers:
            self.options.step_renderers = [TrafficSceneRenderer()]
        renderers = renderers or self.options.step_renderers

        render_params = render_params or RenderParams()
        if render_params.time_step is None:
            render_params.time_step = self.current_time_step
        if render_params.scenario is None:
            render_params.scenario = self.current_scenario
        if render_params.simulation is None:
            render_params.simulation = self

        merged_render_kwargs = render_params.render_kwargs or {}
        merged_render_kwargs.update(self._options.render_kwargs or {})
        merged_render_kwargs.update(render_kwargs or {})
        render_params.render_kwargs = merged_render_kwargs

        frames = [r.render(render_params=render_params, return_frame=return_frames) for r in renderers]
        return frames if return_frames else []

    # Close everything on garbage collection
    def __del__(self) -> None:
        self.close()

    def _sorted_lanelet_indices(
        self,
        lanelet_polylines: List[ContinuousPolyline],
        orientation: float,
        position: np.ndarray
    ) -> List[int]:
        if len(lanelet_polylines) == 0:
            return []
        elif len(lanelet_polylines) == 1:
            return [0]
        else:
            def get_lanelet_sort_value(polyline: ContinuousPolyline) -> float:
                arclength_projection = polyline.get_projected_arclength(
                    position,
                    linear_projection=self.options.linear_lanelet_projection
                )
                lanelet_orientation = polyline.get_direction(arclength_projection)
                relative_arclength_projection = arclength_projection / polyline.length
                orientation_error = abs(relative_orientation(lanelet_orientation, orientation))
                sort_value = 0.0
                if relative_arclength_projection <= 0.0 or relative_arclength_projection >= 1.0:
                    sort_value -= 10.0
                sort_value -= orientation_error
                return sort_value

            sorted_indices = sorted(range(len(lanelet_polylines)),
                                    key=lambda i: get_lanelet_sort_value(lanelet_polylines[i]), reverse=True)
            return sorted_indices
