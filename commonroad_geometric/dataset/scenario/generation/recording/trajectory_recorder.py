import os
import pickle
from typing import Iterable, Optional, Set
from pathlib import Path
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin

from commonroad_geometric.dataset.scenario.generation.recording.sumo_recorder import SumoRecorder, SumoRecordingData
from commonroad_geometric.dataset.scenario.generation.recording.trajectory_generator import TrajectoryGenerator
from commonroad_geometric.dataset.scenario.generation.recording.trajectory_metadata import TrajectoryMetadata, count_crossed_intersections, count_possible_collisions
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation


def _save_scenario(
    output_path: Path,
    scenario: Scenario,
    planning_problem_set: Optional[PlanningProblemSet] = None
) -> Path:
    from commonroad.common.file_writer import CommonRoadFileWriter
    from commonroad.common.file_writer import OverwriteExistingFile

    if planning_problem_set is None:
        planning_problem_set = PlanningProblemSet(planning_problem_list=[])

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    file_writer = CommonRoadFileWriter(scenario=scenario,
                                       planning_problem_set=planning_problem_set)
    try:
        file_writer.write_to_file(
            str(output_path),
            overwrite_existing_file=OverwriteExistingFile.ALWAYS,
        )
    except Exception as e:
        raise ValueError(f"Failed to write to: {output_path}\n{repr(e)}")
    return output_path


class TrajectoryRecorder(AutoReprMixin):
    def __init__(
        self,
        sumo_simulation: SumoSimulation,
    ) -> None:
        self._sumo_recorder = SumoRecorder(sumo_simulation=sumo_simulation)
        self._trajectory_generator = TrajectoryGenerator(self._sumo_recorder)

    def start_simulation(self) -> None:
        self._sumo_recorder.start_simulation()

    def close_simulation(self):
        self._sumo_recorder.sumo_simulation.close()

    # Intentional domain/name change for end user
    @property
    def complete_trajectory_ids(self) -> Set[int]:
        return self._sumo_recorder._inactive_obstacle_ids

    @property
    def all_trajectory_ids(self) -> Set[int]:
        return self._sumo_recorder._inactive_obstacle_ids & self._sumo_recorder._active_obstacle_id_to_obstacle_t_minus_one.keys()

    def record_complete_trajectories(self, complete_trajectory_count: int) -> SumoRecordingData:
        """
        Records and returns trajectory_count complete trajectories, meaning from entering to exiting the road network.

        Args:
            complete_trajectory_count (int): How many trajectories should be recorded

        Returns:
             tuple of:
            - trajectory id to trajectory, including all recorded trajectories
            - set of trajectory ids with complete trajectories
        """
        # Delegate to sumo_recorder for clean interface
        return self._sumo_recorder.record_trajectories_for_time_steps(
            initial_time_step=0,
            time_steps=1,
            complete_trajectory_count=complete_trajectory_count
        )

    def record_trajectories_for_time_steps(
        self,
        time_steps: int,
        initial_time_step: int = 0,
        min_trajectory_length: int = 0,
        complete_trajectory_count: int = 0,
        time_step_cutoff: Optional[int] = None,
    ) -> SumoRecordingData:
        """
        Records starting from initial_time_step (allowing for the simulation to "warm up") for time_steps.
        Discards all trajectories which fall below the given minimum trajectory length.

        Args:
            time_steps (int): How many time-steps should be recorded
            initial_time_step (int): Starting from this time-step
            min_trajectory_length (int): With a minimum trajectory length of
            complete_trajectory_count (Optional[int]): How many complete trajectories should be recorded
            time_step_cutoff: (Optional[int]): Optional limit for time-step, defaults to None, potentially recording forever.

        Returns:
            SumoRecordingData, including trajectory id to trajectory dict, where each trajectory has a minimum trajectory length
        """
        # Delegate to sumo_recorder for clean interface
        return self._sumo_recorder.record_trajectories_for_time_steps(
            initial_time_step=initial_time_step,
            time_steps=time_steps,
            min_trajectory_length=min_trajectory_length,
            complete_trajectory_count=complete_trajectory_count,
            time_step_cutoff=time_step_cutoff,
        )

    def save_ego_trajectory_in_scenario(
        self,
        output_dir: Path,
        ego_trajectory_id: int,
        output_filename: Optional[Path] = None,
        ego_obstacle_id: int = -1,
    ) -> TrajectoryMetadata:
        """
            Saves scenario with trajectory of trajectory_id saved as ego obstacle with ego_obstacle_id to output directory.

        Args:
            output_dir (Path): output directory
            ego_trajectory_id (int): Trajectory to be saved.
            output_filename (Optional[Path]): Optional non-default output filename
            ego_obstacle_id (int): ID for dynamic obstacle in scenario.
        Returns:
            TrajectoryMetadata, including filename of scenario file
        """
        scenario, ego_planning_problem_set = self._trajectory_generator.generate_scenario_with_ego_trajectory(ego_trajectory_id=ego_trajectory_id, ego_obstacle_id=ego_obstacle_id)

        if output_filename is not None:
            output_path = Path(output_dir, f'{output_filename}_{SumoRecorder.RECORDER_VERSION}.xml')
        else:
            output_path = Path(output_dir, f'{str(scenario.scenario_id)}_ego_trajectory_{hash(ego_planning_problem_set)}_{SumoRecorder.RECORDER_VERSION}.xml')

        scenario_file_path = _save_scenario(output_path=output_path,
                                            scenario=scenario,
                                            planning_problem_set=ego_planning_problem_set)

        return TrajectoryMetadata(
            scenario_file_path=scenario_file_path,
            trajectory_file_path=None,
            num_vehicles=len(scenario.dynamic_obstacles),
            ego_trajectory_length=len(self._sumo_recorder.trajectory_id_to_trajectory[ego_trajectory_id]),
            crossed_intersections=count_crossed_intersections(lanelet_network=scenario.lanelet_network,
                                                              trajectory_id=ego_trajectory_id,
                                                              trajectory_id_to_trajectory=self._sumo_recorder.trajectory_id_to_trajectory),
            max_possible_collisions=0
        )

    def save_trajectories_in_scenario(
        self,
        scenario_output_dir: Path,
        trajectory_ids: Iterable[int],
        output_filename: Optional[Path] = None,
        ego_trajectory_id: Optional[int] = None,
        ego_obstacle_id: int = -1
    ) -> TrajectoryMetadata:
        """
            Saves scenario with trajectories saved as dynamic obstacles to output directory.

        Args:
            scenario_output_dir (Path): output directory
            trajectory_ids (Iterable[int]): All trajectories to be saved.
            output_filename (Optional[str]): Optional non-default output filename
            ego_trajectory_id (Optional[int]): Trajectory which should be assigned to the ego obstacle.
            ego_obstacle_id (int): ID for ego dynamic obstacle in scenario.

        Returns:
            TrajectoryMetadata, including filename of scenario file
        """
        scenario, ego_planning_problem_set = self._trajectory_generator.generate_scenario_with_trajectories(trajectory_ids=trajectory_ids,
                                                                                                            ego_trajectory_id=ego_trajectory_id,
                                                                                                            ego_obstacle_id=ego_obstacle_id)
        if output_filename is not None:
            output_path = scenario_output_dir / f"{output_filename}_{SumoRecorder.RECORDER_VERSION}.xml"
        else:
            output_path = Path(scenario_output_dir,
                               f"{str(scenario.scenario_id)}"
                               f"_trajectories_{hash(tuple(scenario.dynamic_obstacles))}"
                               f"_{SumoRecorder.RECORDER_VERSION}.xml")

        scenario_file_path = _save_scenario(output_path=output_path,
                                            scenario=scenario,
                                            planning_problem_set=ego_planning_problem_set)

        trajectory_recording_metadata = TrajectoryMetadata(
            scenario_file_path=scenario_file_path,
            trajectory_file_path=None,
            num_vehicles=len(scenario.dynamic_obstacles)
        )

        if ego_trajectory_id is not None:
            trajectory_recording_metadata.ego_trajectory_length = len(
                self._sumo_recorder.trajectory_id_to_trajectory[ego_trajectory_id])
            trajectory_recording_metadata.crossed_intersections = count_crossed_intersections(
                lanelet_network=scenario.lanelet_network,
                trajectory_id=ego_trajectory_id,
                trajectory_id_to_trajectory=self._sumo_recorder.trajectory_id_to_trajectory
            )
            trajectory_recording_metadata.max_possible_collisions = count_possible_collisions(
                trajectory_id=ego_trajectory_id,
                trajectory_id_to_trajectory=self._sumo_recorder.trajectory_id_to_trajectory
            )

        return trajectory_recording_metadata
