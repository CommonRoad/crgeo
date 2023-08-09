from __future__ import annotations

import copy
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple, Type, Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker
from commonroad_dc.pycrcc import CollisionChecker  # noqa
from traci.exceptions import FatalTraCIError

from commonroad_geometric.common.io_extensions.scenario import backup_scenario, get_dynamic_obstacles_at_timestep
from commonroad_geometric.common.logging import stdout, stdout_clear
from commonroad_geometric.dataset.extraction.road_network.types import GraphConversionStep
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.base_simulation import BaseSimulation, BaseSimulationOptions, Unlimited
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.simulation.exceptions import SimulationRuntimeError
from commonroad_geometric.simulation.interfaces.interactive._cr_sumo_simulation import _CRSumoSimulation
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation_config import DefaultSumoConfig
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning import BaseTrafficSpawner
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantRateSpawner

logger = logging.getLogger(__name__)

MAX_AUTO_PRESIMULATION_STEPS = 1000


MAX_PRESIMULATION_STEPS = 1500


class SumoPresimulationError(SimulationRuntimeError):
    pass


class PreSimulationStepLimitExceededError(SumoPresimulationError):
    pass


@dataclass
class SumoSimulationOptions(BaseSimulationOptions):
    """
        presimulation_steps (int): Silent presimulation steps run within SUMO without updating state of this class.
        traffic_spawner (BaseTrafficSpawner): Traffic spawner instance. Defaults to ConstantRateSpawner.
        p_wants_lane_change (float): Probability of spawned vehicles wanting to change their lane
        p_car (float): Probability of a car being the newly spawned vehicle
        p_truck (float): Probability of a truck being the newly spawned vehicle
        p_bus (float): Probability of a bus spawning being the newly spawned vehicle
    """
    presimulation_steps: Union[int, Literal['auto']] = 'auto'
    traffic_spawner: BaseTrafficSpawner = ConstantRateSpawner()
    init_vehicle_speed: float = 5.0
    max_vehicle_speed: float = 10.0
    p_wants_lane_change: float = 0.25
    p_car: float = 1.0
    p_truck: float = 0.0
    p_bus: float = 0.0
    inactive_restart_threshold: float = 5.0
    ego_planning_problem_set: Optional[PlanningProblemSet] = None


class SumoSimulation(BaseSimulation):
    def __init__(
        self,
        initial_scenario: Union[Scenario, str],
        options: Optional[SumoSimulationOptions] = None,
        report_presimulation_progress: bool = True,
    ) -> None:
        """
        Simulation using the traffic simulator SUMO as backend.

        Args:
            initial_scenario (Union[Scenario, str]): Initial scenario for this simulation.
            options (SumoSimulationOptions): Options for SUMO and this simulation.
            dt (float): Positive time delta between each time-step. Integral granularity.
        """
        self._options: SumoSimulationOptions = options or SumoSimulationOptions()
        self._report_presimulation_progress = report_presimulation_progress
        self._sumo_sim = _CRSumoSimulation()
        self._sumo_conf = DefaultSumoConfig()
        self._sumo_conf.dt = self._options.dt
        self._traffic_spawner = self._options.traffic_spawner
        self._sumo_conf.veh_distribution[ObstacleType.CAR] = self._options.p_car
        self._sumo_conf.veh_distribution[ObstacleType.TRUCK] = self._options.p_truck
        self._sumo_conf.veh_distribution[ObstacleType.BUS] = self._options.p_bus

        super().__init__(
            initial_scenario=initial_scenario,
            options=options
        )

        # Remove all obstacles that might be still in the scenario as SUMO will spawn obstacle vehicles
        for obstacle in self.initial_scenario.dynamic_obstacles:
            self.initial_scenario.remove_obstacle(obstacle)
        for obstacle in self.current_scenario.dynamic_obstacles:
            self.current_scenario.remove_obstacle(obstacle)

        if self._options.collision_checking:
            # Create empty collision checker if no step has been run, and no vehicles have spawned yet
            self._current_collision_checker = create_collision_checker(self.current_scenario)
        # For discovering disconnect from SUMO
        self._last_step_timestamp: Optional[float] = None

    @BaseSimulation.current_time_step.setter
    def current_time_step(self, time_step_to_jump_to: int) -> None:
        raise SimulationRuntimeError(f"BaseSimulation.current_time_step should never be set directly for {type(self).__name__}.")

    @property
    def collision_checker(self) -> CollisionChecker:
        return self._current_collision_checker

    def _get_time_step_bounds(self) -> Tuple[int, Type[Unlimited]]:
        return 0, Unlimited

    def _step(
        self,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> None:
        if ego_vehicle is not None:
            trajectory_ego = [copy.deepcopy(ego_vehicle.state)]
            trajectory_ego[0].time_step = 1
            if not self._sumo_sim.has_ego_vehicle:
                self._sumo_sim.init_ego_vehicle(ego_vehicle.ego_route.planning_problem)
            self._sumo_sim.ego_vehicle_first.set_planned_trajectory(trajectory_ego)
            self._sumo_sim.forward_info2sumo(
                planned_state=ego_vehicle.state,
                sync_mechanism='SYNC_MOVE_XY',
                lc_duration=0
            )

        self.spawn_vehicles()
        self._sumo_sim.simulate_step()

        # This call fails if start_0 is set to anything but True (default), which is done here explicitly for clarity
        # Reason being bound-checking code in trajectory.py
        # There is no obvious function to set the correct time-step and avoid this issue
        self._current_scenario = self._sumo_sim.commonroad_scenario_at_time_step(
            time_step=self._sumo_sim.current_time_step,
            add_ego=False,
            start_0=True
        )

        # Hence, we set the correct time-steps for all dynamic obstacles after the fact, avoiding issues down the line
        for obstacle in self.current_scenario.dynamic_obstacles:
            # Not sure about this one
            obstacle.initial_state.time_step = self.current_time_step
            obstacle.prediction.initial_time_step = self.current_time_step
            # Very sure about this one
            obstacle.prediction.final_time_step = self.current_time_step

        # Create collision checker before adding ego vehicle
        self._current_collision_checker = create_collision_checker(self.current_scenario)

        # Re-add ego-vehicle for the scenario which was newly created from SUMO, i.e. the new current scenario from above
        # Add it after creating the collision checker, otherwise it will collide with itself
        if ego_vehicle is not None and ego_vehicle not in self.current_scenario.dynamic_obstacles:
            self.current_scenario.add_objects(ego_vehicle.as_dynamic_obstacle)

        self._time_step_to_obstacles[self.current_time_step] = get_dynamic_obstacles_at_timestep(self.current_scenario, self.current_time_step)

    def _spawn_ego_vehicle(self, ego_vehicle: EgoVehicle) -> None:
        # HACK to spawn ego vehicle within SUMO.
        # For self_initialize_sumo no planning_problem_set is available.
        # We only have to spawn the ego vehicle once for it to be included in the SUMO simulation
        #
        # self._sumo_sim.ego_vehicle_first.set_planned_trajectory(trajectory_ego) will then update its state
        # and other vehicles will see it/ try not to collide with it
        #
        # The planning_problem_set passed here doesn't really matter i.e. we can respawn with random planning problems without issues
        # init_ego_vehicles_from_planning_problem calls presimulation_silent(1), which only works at _CRSumoSimulation.current_time_step == 0
        if not self._sumo_sim.ego_vehicles and self._sumo_sim.current_time_step == 0:
            self._sumo_sim.init_ego_vehicle(ego_vehicle.ego_route.planning_problem)
            logger.debug(f"Initialized ego vehicle with ID {ego_vehicle.obstacle_id} in SUMO")
        logger.debug(f"Spawned ego vehicle with ID {ego_vehicle.obstacle_id} in SUMO")
        # Run next step with ego vehicle, otherwise ego vehicle will not be added to current_scenario in SumoSimulation._step
        # even though it has spawned just now
        self.lifecycle.run_ego()
        next(self)

    def _start(self) -> None:
        if not self._sumo_sim.connected:
            self._initialize_sumo()
        self._traffic_spawner.reset(self.current_scenario)
        if self._options.collision_checking:
            # Create initial collision checker after possible presimulation
            self._current_collision_checker = create_collision_checker(self.current_scenario)

    def _close(self) -> None:
        try:
            self._sumo_sim.stop()
        except FatalTraCIError as e:
            warnings.warn(f"Failed to stop SUMO simulation: {repr(e)}")

    def _reset(self) -> None:
        self._current_collision_checker = None
        # Re-establish connection to SUMO if it has been dropped
        seconds_since_last_step = time.time() - self._last_step_timestamp if self._last_step_timestamp is not None else 0.0
        if seconds_since_last_step > self._options.inactive_restart_threshold:
            logger.info(f"Reconnecting SUMO due to inactivity ({seconds_since_last_step:.1f} seconds)...")
            reconnect_sumo = True
        elif not self._sumo_sim.connected:
            logger.info("Connection to SUMO lost, reconnecting...")
            reconnect_sumo = True
        else:
            reconnect_sumo = False
        if reconnect_sumo:
            if self._sumo_sim.connected:
                self._sumo_sim.stop()
            self._initialize_sumo()

        # SUMO specific reset of BaseSimulation state
        if self._options.backup_initial_scenario:
            self._current_scenario = backup_scenario(self.initial_scenario)
        if self._options.collision_checking:
            self._current_collision_checker = create_collision_checker(self.current_scenario)
        self._traffic_spawner.reset(self.current_scenario)
        self._time_step_to_obstacles = defaultdict(list)

    def _initialize_sumo(self) -> None:
        logger.debug("Initializing SUMO simulation")
        self._sumo_sim.initialize(
            conf=self._sumo_conf,
            scenario_wrapper=self.current_scenario,
            planning_problem_set=None,  # This will spawn ego vehicles for every planning problem in the planning problem set
            presimulation_steps=0
        )
        self.presimulation_silent(self._options.presimulation_steps)

    def spawn_vehicles(self) -> int:
        to_spawn = self._traffic_spawner.spawn(
            dt=self.dt,
            time_step=self.current_time_step,
            scenario=self.current_scenario
        )
        for lanelet_id in to_spawn:
            self._sumo_sim.spawn_vehicle(
                init_speed=self._options.init_vehicle_speed,
                max_speed=self._options.max_vehicle_speed,
                p_wants_lane_change=self._options.p_wants_lane_change,
                lanelet_id=lanelet_id
            )
        return len(to_spawn)

    def presimulation_silent(self, pre_simulation_steps: Union[int, Literal['auto']]) -> int:
        """
        Simulate SUMO without synchronization of interface. Used before starting interface simulation in order to populate scenario.

        :param pre_simulation_steps: the steps of simulation which are executed. If 'auto' is provided, it will continue until the first vehicle exits the scenario.

        """

        logger.debug(f"Starting SUMO presimulation with option {pre_simulation_steps}")

        if self._current_time_step is None:
            self._current_time_step = 0
        start_time_step = self.current_time_step

        steps = 0
        self._traffic_spawner.reset(self.current_scenario)

        def presimulation_step() -> Scenario:
            assert self._current_time_step is not None
            nonlocal steps
            self._sumo_sim.simulate_step(silent=False)
            scenario: Scenario = self._sumo_sim.commonroad_scenario_at_time_step(
                time_step=self._sumo_sim.current_time_step,
                add_ego=False,
                start_0=True
            )
            num_spawned_vehicle = self.spawn_vehicles()
            if num_spawned_vehicle > 0:
                logger.debug(f"Spawned {num_spawned_vehicle} new vehicles at presimulation step {steps}")

            if self._options.step_renderers:
                self.render(render_params=RenderParams(
                    scenario=scenario,
                    time_step=start_time_step,
                    simulation=self,
                    render_kwargs=dict(
                        overlays={
                            'Presimulation-step:': steps
                        }
                    )
                ))

            self._current_time_step += 1
            steps += 1
            if self._report_presimulation_progress:
                stdout(f"SUMO presimulation step: {steps} ({len(self._sumo_sim.current_vehicle_ids)} vehicles present)")

            if steps > MAX_PRESIMULATION_STEPS:
                raise SimulationRuntimeError(f"SUMO presimulation steps exceeded upper bound {MAX_PRESIMULATION_STEPS}")

            return scenario

        if isinstance(pre_simulation_steps, int):
            for _ in range(pre_simulation_steps):
                self._current_scenario = presimulation_step()

        elif pre_simulation_steps == "auto":
            last_vehicle_ids: Set[int] = set()
            for _ in range(MAX_AUTO_PRESIMULATION_STEPS):
                self._current_scenario = presimulation_step()
                current_vehicle_ids = {obstacle.obstacle_id for obstacle in self._current_scenario.dynamic_obstacles}
                removed_vehicles_ids = last_vehicle_ids - current_vehicle_ids
                if len(removed_vehicles_ids) > 0:
                    break
                last_vehicle_ids = current_vehicle_ids
            else:
                raise PreSimulationStepLimitExceededError(f"Exceeded limit of {MAX_AUTO_PRESIMULATION_STEPS} pre-simulation steps")

        else:
            raise ValueError(f"Unknown value for pre_simulation_steps argument: {pre_simulation_steps}")

        if self._report_presimulation_progress:
            stdout_clear()
        self._current_time_step = start_time_step

        logger.debug(f"Presimulation with option {pre_simulation_steps} completed, took {steps} steps")

        return steps
