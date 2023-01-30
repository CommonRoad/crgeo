import sys, os; sys.path.insert(0, os.getcwd())

from typing import List

from crgeo.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions
from crgeo.simulation.interfaces.static.compressed_scenario_simulation import CompressedScenarioSimulation, CompressedSimulationOptions
from crgeo.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions
from crgeo.dataset.generation.recording import TrajectoryRecorder, TrajectoryMetadata
from crgeo.rendering.traffic_scene_renderer import TrafficSceneRenderer


TRAJECTORY_COUNT = 10
INPUT_SCENARIO = 'data/other/USA_US101-26_1_T-1.xml'
OUTPUT_DIR = 'tutorials/output/trajectories'


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


def record_scenario_pickle_trajectories(sumo_options: SumoSimulationOptions) -> TrajectoryMetadata:
    simulation = SumoSimulation(
        initial_scenario=INPUT_SCENARIO,
        options=sumo_options
    )
    trajectory_recorder = TrajectoryRecorder(sumo_simulation=simulation)
    trajectory_recorder.start_simulation()
    sumo_recording_data = trajectory_recorder.record_trajectories_for_time_steps(time_steps=500,
                                                                                 min_trajectory_length=100)
    trajectory_metadata = trajectory_recorder.save_scenario_pickle_trajectories(scenario_output_dir=OUTPUT_DIR,
                                                                                pickle_output_dir=OUTPUT_DIR,
                                                                                trajectory_ids=sumo_recording_data.trajectory_id_to_trajectory.keys())
    print(f"Scenario with trajectories stubs written to: {trajectory_metadata.scenario_file_path}")
    print(f"Dictionary of trajectory ids written to: {trajectory_metadata.trajectory_file_path}")
    trajectory_recorder.close_simulation()
    return trajectory_metadata


def replay_scenario(input_scenario: str, scenario_options: ScenarioSimulationOptions) -> None:
    simulation = ScenarioSimulation(initial_scenario=input_scenario,
                                    options=scenario_options)
    simulation.start()
    for time_step, scenario in simulation:
        pass
    simulation.close()


def replay_compressed_scenario(trajectory_metadata: TrajectoryMetadata, scenario_options: CompressedSimulationOptions) -> None:
    scenario_options.trajectory_pickle_file = trajectory_metadata.trajectory_file_path
    compressed_scenario_simulation = CompressedScenarioSimulation(initial_scenario=trajectory_metadata.scenario_file_path,
                                                                  options=scenario_options)
    compressed_scenario_simulation.start()
    for time_step, scenario in compressed_scenario_simulation:
        pass
    compressed_scenario_simulation.close()

    # The two simulations need to render to different renderers
    scenario_simulation_options = ScenarioSimulationOptions(step_renderers=TrafficSceneRenderer())
    # After creating a CompressedScenarioSimulation we can create a ScenarioSimulation from its reconstructed scenario
    scenario_simulation = ScenarioSimulation(initial_scenario=compressed_scenario_simulation.initial_scenario,
                                             options=scenario_simulation_options)

    # We can rerun CompressedScenarioSimulation as many times as we want
    compressed_scenario_simulation.start()
    scenario_simulation.start()

    # Run both at the same time to check that they are the same
    for (time_step, scenario), (c_time_step, c_scenario) in zip(scenario_simulation, compressed_scenario_simulation):
        assert time_step == c_time_step
    compressed_scenario_simulation.close()
    scenario_simulation.close()


if __name__ == '__main__':
    renderer = TrafficSceneRenderer()
    _sumo_options = SumoSimulationOptions(step_renderers=renderer,
                                          presimulation_steps=0)
    ego_trajectory_metadata = record_ego_trajectories(_sumo_options)

    _scenario_options = CompressedSimulationOptions(step_renderers=TrafficSceneRenderer())
    for ego_metadata in ego_trajectory_metadata:
        print(ego_metadata)
        replay_scenario(input_scenario=ego_metadata.scenario_file_path, scenario_options=_scenario_options)

    _trajectory_metadata = record_scenario_trajectories(_sumo_options)
    replay_scenario(input_scenario=_trajectory_metadata.scenario_file_path, scenario_options=_scenario_options)

    compressed_trajectory_metadata = record_scenario_pickle_trajectories(_sumo_options)
    replay_compressed_scenario(trajectory_metadata=compressed_trajectory_metadata, scenario_options=_scenario_options)
