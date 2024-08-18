from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from tqdm import tqdm

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State, CustomState
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.io_extensions.scenario import get_dynamic_obstacles_at_timesteps
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation


@dataclass
class SumoRecordingData:
    trajectory_id_to_trajectory: Dict[int, List[State]]
    complete_trajectory_ids: Set[int]
    num_vehicles: int


class SumoRecorder(AutoReprMixin):
    RECORDER_VERSION = 'V1_0'
    DEFAULT_ATTRIBUTES = ['position', 'orientation', 'velocity', 'acceleration']

    def __init__(
        self,
        sumo_simulation: SumoSimulation,
        recorded_attributes: Optional[List[str]] = None
    ) -> None:
        """
        Recorder for trajectories of the given SUMO simulation.
        Can record complete trajectories of obstacle vehicles, i.e. from entering to leaving road network.
        Can record trajectories with minimum length for a certain amount of time-steps, i.e. during any time of their lifespan.

        Args:
            sumo_simulation (SumoSimulation): SumoSimulation of the SUMO simulation that should be recorded.
            recorded_attributes (Optional[List[str]]): Optional list of attributes which should be recorded.
        """
        self._simulation: SumoSimulation = sumo_simulation
        self._recorded_attributes = recorded_attributes or SumoRecorder.DEFAULT_ATTRIBUTES

        # Contains initial_time_step for respective obstacle_id, important for generating scenario with all trajectories
        # as not all trajectories start at time_step 0
        self._obstacle_id_to_obstacle_in_initial_state: Dict[int, DynamicObstacle] = {}
        self._obstacle_id_to_obstacle_in_final_state: Dict[int, DynamicObstacle] = {}
        # Allows accessing of final_state by looking into past after obstacle left road network
        self._active_obstacle_id_to_obstacle_t_minus_one: Dict[int, DynamicObstacle] = {}
        self._inactive_obstacle_ids: Set[int] = set()
        self._obstacle_id_to_trajectory: Dict[int, List[State]] = defaultdict(list)

    @property
    def sumo_simulation(self) -> SumoSimulation:
        return self._simulation

    @property
    def recorded_scenario(self):
        return self._simulation.initial_scenario

    @property
    def recorded_attributes(self):
        return self._recorded_attributes

    def obstacle_in_initial_state(self, obstacle_id: int) -> Optional[DynamicObstacle]:
        if obstacle_id in self._obstacle_id_to_obstacle_in_initial_state:
            return self._obstacle_id_to_obstacle_in_initial_state[obstacle_id]
        return None

    def obstacle_in_final_state(self, obstacle_id: int) -> Optional[DynamicObstacle]:
        if obstacle_id in self._obstacle_id_to_obstacle_in_final_state:
            return self._obstacle_id_to_obstacle_in_final_state[obstacle_id]
        return None

    # Intentional domain/name change for end user
    @property
    def trajectory_id_to_trajectory(self) -> Dict[int, List[State]]:
        return self._obstacle_id_to_trajectory

    def start_simulation(self) -> None:
        self._simulation.start()

    def record_trajectories_for_time_steps(
        self,
        time_steps: int,
        initial_time_step: int = 0,
        min_trajectory_length: int = 0,
        complete_trajectory_count: int = 0,
        time_step_cutoff: Optional[int] = None,
        render_generation: bool = True
    ) -> SumoRecordingData:
        """
        Records starting from initial_time_step (allowing for the simulation to "warm up") for time_steps.
        Discards all trajectories which fall below the given minimum trajectory length.
        Optionally records and returns complete_trajectory_count complete trajectories, meaning from entering to exiting the road network.
        This supersedes the time_steps argument, recording until complete_trajectory_count is reached.

        Args:
            time_steps (int): How many time-steps should be recorded
            initial_time_step (int): Starting from this time-step, defaults to 0
            min_trajectory_length (int): With a minimum trajectory length of, defaults to 0
            complete_trajectory_count (int): How many complete trajectories should be recorded, defaults to 0
            time_step_cutoff: (Optional[int]): Optional limit for time-step, defaults to None, potentially recording forever.

        Returns:
            tuple of:
            - trajectory id to trajectory, where each trajectory has a minimum trajectory length
            - set of trajectory ids with complete trajectories
        """
        self._inactive_obstacle_ids = set()
        self._obstacle_id_to_trajectory = defaultdict(list)

        final_time_step = initial_time_step + time_steps
        time_step_cutoff = max(time_step_cutoff if time_step_cutoff is not None else 0, final_time_step)
        # Don't record the steps before the initial_time_step, this is TODO supposed to be handled by the SumoSimulation internally
        # Don't pass to_time_step argument as we want to ensure that the
        # simulation can run until complete_trajectory_count is reached
        for time_step, scenario in tqdm(self._simulation):
            if time_step < initial_time_step:
                continue
            if time_step > final_time_step:
                if len(self._inactive_obstacle_ids) >= complete_trajectory_count:
                    break
                if time_step > time_step_cutoff:
                    break
            newly_inactive_obstacles = self.record_step(time_step, scenario)
            self._inactive_obstacle_ids |= newly_inactive_obstacles

            if render_generation:
                self._simulation.render()

        # Add missing final states for active obstacles
        for obstacle_id, obstacle in self._active_obstacle_id_to_obstacle_t_minus_one.items():
            self._obstacle_id_to_obstacle_in_final_state[obstacle_id] = obstacle

        trajectory_id_to_trajectory = {}
        for obstacle_id, trajectory in self._obstacle_id_to_trajectory.items():
            if len(trajectory) > min_trajectory_length:
                trajectory_id_to_trajectory[obstacle_id] = trajectory
        return SumoRecordingData(
            trajectory_id_to_trajectory=trajectory_id_to_trajectory,
            complete_trajectory_ids=self._inactive_obstacle_ids,
            num_vehicles=len(self.trajectory_id_to_trajectory)
        )

    def record_step(
        self,
        time_step: int,
        scenario: Scenario
    ) -> Set[int]:
        """
        Records one step of the SUMO simulation internally in _obstacle_id_to_trajectory.
        Sets _active_obstacle_id_to_obstacle_t_minus_one, the active vehicles for the step that was recorded.
        Sets _obstacle_id_to_obstacle_initial_state, the initial/first recorded dynamic obstacle in its initial state.
        Sets _obstacle_id_to_obstacle_in_final_state for an obstacle in its final state after it becomes inactive.

        Returns:
            Newly inactive obstacle ids, i.e. the vehicles which just left the SUMO simulation at the edge of the map.
        """
        obstacles = get_dynamic_obstacles_at_timesteps(scenario)[time_step]

        active_obstacle_id_to_obstacle = {}
        for obstacle in obstacles:
            # Save the initial state of a vehicle, vehicle either just spawned or we just started recording
            if obstacle.obstacle_id not in self._active_obstacle_id_to_obstacle_t_minus_one.keys():
                self._obstacle_id_to_obstacle_in_initial_state[obstacle.obstacle_id] = obstacle
            trajectory_sample = self._get_trajectory_sample(obstacle)
            active_obstacle_id_to_obstacle[obstacle.obstacle_id] = obstacle
            self._obstacle_id_to_trajectory[obstacle.obstacle_id].append(trajectory_sample)

        # Track final state used for GoalRegion attribute, go back in time one step before it left the road network
        inactive_obstacle_ids = self._active_obstacle_id_to_obstacle_t_minus_one.keys() - active_obstacle_id_to_obstacle.keys()
        for obstacle in self._active_obstacle_id_to_obstacle_t_minus_one.values():
            if obstacle.obstacle_id in inactive_obstacle_ids:
                self._obstacle_id_to_obstacle_in_final_state[obstacle.obstacle_id] = obstacle

        self._active_obstacle_id_to_obstacle_t_minus_one = active_obstacle_id_to_obstacle
        return inactive_obstacle_ids

    def _get_trajectory_sample(self, obstacle: DynamicObstacle) -> State:
        obstacle_state = state_at_time(obstacle, self._simulation.current_time_step, assume_valid=True)
        trajectory_attributes = {attribute: getattr(obstacle_state, attribute)
                                 for attribute in self._recorded_attributes}
        trajectory_sample = CustomState(time_step=self._simulation.current_time_step, **trajectory_attributes)
        return trajectory_sample
