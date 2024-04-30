import sys; import os; sys.path.insert(0, os.getcwd())

from typing import List
from pathlib import Path
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from commonroad_geometric.dataset.scenario.generation.recording import TrajectoryRecorder, TrajectoryMetadata
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer

TRAJECTORY_COUNT = 10
INPUT_SCENARIO = Path('data/other/USA_US101-26_1_T-1.xml')
OUTPUT_DIR = Path('tutorials/output/trajectories')


def record_ego_trajectories(sumo_options: SumoSimulationOptions) -> List[TrajectoryMetadata]:
    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=sumo_options
    )
    trajectory_recorder = TrajectoryRecorder(sumo_simulation=simulation)
    trajectory_recorder.start_simulation()
    sumo_recording_data = trajectory_recorder.record_complete_trajectories(complete_trajectory_count=TRAJECTORY_COUNT)
    print(f"Recorded {len(sumo_recording_data.trajectory_id_to_trajectory)} trajectories: ")
    output_paths = []
    for trajectory_id in sumo_recording_data.complete_trajectory_ids:
        trajectory = sumo_recording_data.trajectory_id_to_trajectory[trajectory_id]
        print(f" - Trajectory id {trajectory_id} length: {len(trajectory)}")
        output_path = trajectory_recorder.save_ego_trajectory_in_scenario(output_dir=OUTPUT_DIR,
                                                                          ego_trajectory_id=trajectory_id)
        output_paths.append(output_path)
        print(f"Scenario with ego trajectory {trajectory_id} written to: {output_path}")
    trajectory_recorder.close_simulation()
    return output_paths


def record_scenario_trajectories(sumo_options: SumoSimulationOptions) -> TrajectoryMetadata:
    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=sumo_options
    )
    trajectory_recorder = TrajectoryRecorder(sumo_simulation=simulation)
    trajectory_recorder.start_simulation()
    sumo_recording_data = trajectory_recorder.record_trajectories_for_time_steps(time_steps=500,
                                                                                 min_trajectory_length=100)

    ego_trajectory_id = next(iter(sumo_recording_data.trajectory_id_to_trajectory))

    output_path = trajectory_recorder.save_trajectories_in_scenario(scenario_output_dir=OUTPUT_DIR,
                                                                    trajectory_ids=sumo_recording_data.trajectory_id_to_trajectory.keys(),
                                                                    ego_trajectory_id=ego_trajectory_id)
    print(f"Scenario with all trajectories written to: {output_path}")
    trajectory_recorder.close_simulation()
    return output_path


def replay_scenario(input_scenario: Path, scenario_options: ScenarioSimulationOptions) -> None:
    simulation = ScenarioSimulation(initial_scenario=input_scenario,
                                    options=scenario_options)
    simulation.start()
    for time_step, scenario in simulation:
        pass
    simulation.close()


if __name__ == '__main__':
    renderer = TrafficSceneRenderer()
    _sumo_options = SumoSimulationOptions(step_renderers=[renderer],
                                          presimulation_steps=0)
    ego_trajectory_metadata = record_ego_trajectories(_sumo_options)

    _scenario_options = ScenarioSimulationOptions(step_renderers=[renderer])
    for ego_metadata in ego_trajectory_metadata:
        print(ego_metadata)
        replay_scenario(input_scenario=ego_metadata.scenario_file_path, scenario_options=_scenario_options)

    _trajectory_metadata = record_scenario_trajectories(_sumo_options)
    replay_scenario(input_scenario=_trajectory_metadata.scenario_file_path, scenario_options=_scenario_options)
