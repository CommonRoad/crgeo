import json
import logging
import os
import random
import shutil
import sys
from dataclasses import asdict
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Union

from commonroad.scenario.scenario import Scenario

from commonroad_geometric.dataset.scenario.generation.recording import TrajectoryMetadata, TrajectoryRecorder
from commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulation, SumoSimulationOptions

logger = logging.getLogger(__name__)


def error_callback(err) -> None:
    logger.error(msg=err, exc_info=True)


def mute_process() -> None:
    sys.stdout = open(os.devnull, 'w')


def generate_traffic(
    input_scenario_iterator: ScenarioIterator,
    sumo_simulation_options: SumoSimulationOptions,
    scenario_output_dir: Path,
    metadata_log_path: Optional[str] = None,
    num_workers: int = 1,
    should_mute_workers: bool = False,
    time_steps_per_run: int = 1000,
    initial_recorded_time_step: int = 0,
    min_trajectory_length: int = 0,
    complete_trajectory_count: int = 0,
    include_ego_vehicle_trajectory: bool = False,
    time_step_cutoff: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Optional[TrajectoryMetadata]]:

    if overwrite:
        shutil.rmtree(scenario_output_dir, ignore_errors=True)

    logger.info(f"Generating traffic for {len(input_scenario_iterator.scenario_paths)} maps")

    scenario_id_to_metadata: Dict[str, Optional[TrajectoryMetadata]] = {}
    if num_workers == 1:
        for scenario_bundle in input_scenario_iterator:
            trajectory_recording_metadata = _generate_traffic(
                scenario_bundle=scenario_bundle,
                sumo_simulation_options=sumo_simulation_options,
                scenario_output_dir=scenario_output_dir,
                time_steps_per_run=time_steps_per_run,
                initial_recorded_time_step=initial_recorded_time_step,
                min_trajectory_length=min_trajectory_length,
                complete_trajectory_count=complete_trajectory_count,
                include_ego_vehicle_trajectory=include_ego_vehicle_trajectory,
                time_step_cutoff=time_step_cutoff,
            )
            scenario_id_to_metadata[str(scenario_bundle.preprocessed_scenario.scenario_id)
                                    ] = trajectory_recording_metadata
    else:
        with Pool(processes=num_workers, initializer=mute_process if should_mute_workers else None) as pool:
            func_args_iterable = zip(  # TODO: should not rely on argument order
                input_scenario_iterator,
                repeat(sumo_simulation_options),
                repeat(scenario_output_dir),
                repeat(time_steps_per_run),
                repeat(initial_recorded_time_step),
                repeat(min_trajectory_length),
                repeat(complete_trajectory_count),
                repeat(include_ego_vehicle_trajectory),
                repeat(time_step_cutoff)
            )
            results = {}
            for func_args in func_args_iterable:
                input_scenario_bundle: ScenarioBundle = func_args[0]
                result = pool.apply_async(
                    func=_generate_traffic,
                    args=func_args,
                    error_callback=error_callback
                )
                results[str(input_scenario_bundle.preprocessed_scenario.scenario_id)] = result
            for scenario_id, result in results.items():
                result.wait()
                if result.successful():  # Result is successful even if _generate_traffic returns None
                    trajectory_recording_metadata = result.get()
                    scenario_id_to_metadata[scenario_id] = trajectory_recording_metadata
        pool.close()
        pool.join()

    if metadata_log_path is not None:
        metadata_log = []
        for metadata in scenario_id_to_metadata.values():
            if metadata is not None:
                metadata_dict = asdict(metadata)
                metadata_dict_filtered = {}
                for k, v in metadata_dict.items():
                    if v is None:
                        continue
                    if isinstance(v, Path):
                        metadata_dict_filtered[k] = str(v)
                    else:
                        metadata_dict_filtered[k] = v
                metadata_log.append(metadata_dict_filtered)

        output_dir = os.path.dirname(metadata_log_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(metadata_log_path, 'w') as metadata_log_json:
            logger.info(
                f"Traffic generation metadata log saved to {metadata_log_path} containing {len(metadata_log)} log entries")
            json.dump(metadata_log, metadata_log_json, sort_keys=True, indent=4)
    return scenario_id_to_metadata


def _generate_traffic(
    scenario_bundle: ScenarioBundle,
    sumo_simulation_options: SumoSimulationOptions,
    scenario_output_dir: Union[Path, str],
    time_steps_per_run: int = 100,
    initial_recorded_time_step: int = 0,
    min_trajectory_length: int = 0,
    complete_trajectory_count: int = 0,
    include_ego_vehicle_trajectory: bool = False,
    time_step_cutoff: Optional[int] = None
) -> Optional[TrajectoryMetadata]:
    logger.info(
        f"Process {os.getpid()} with parent process {os.getppid()} is generating traffic for scenario {scenario_bundle.preprocessed_scenario.scenario_id}")
    trajectory_recorder: Optional[TrajectoryRecorder] = None
    # Putting _create_trajectory_recorder outside the try-block leads to uncaught crashes
    try:
        trajectory_recorder = _create_trajectory_recorder(
            scenario_bundle.preprocessed_scenario, sumo_simulation_options)
        trajectory_recorder.start_simulation()
        sumo_recording_data = trajectory_recorder.record_trajectories_for_time_steps(
            time_steps=time_steps_per_run,
            initial_time_step=initial_recorded_time_step,
            min_trajectory_length=min_trajectory_length,
            complete_trajectory_count=complete_trajectory_count,
            time_step_cutoff=time_step_cutoff,
        )

        ego_trajectory_id = None
        if include_ego_vehicle_trajectory:
            if sumo_recording_data.complete_trajectory_ids:
                ego_trajectory_id = random.choice(list(sumo_recording_data.complete_trajectory_ids))
            else:
                ego_trajectory_id = random.choice(list(sumo_recording_data.trajectory_id_to_trajectory.keys()))
        ego_included = "_includes_ego" if include_ego_vehicle_trajectory and ego_trajectory_id is not None else ""

        logger.info(
            f"Collected {sumo_recording_data.num_vehicles} trajectories for map {scenario_bundle.preprocessed_scenario.scenario_id}")

        filename = (f"{scenario_bundle.preprocessed_scenario.scenario_id}"
                    f"_{hash(scenario_bundle)}"
                    f"_time_steps_{time_steps_per_run}{ego_included}")
        return trajectory_recorder.save_trajectories_in_scenario(
            scenario_output_dir=scenario_output_dir,
            output_filename=Path(filename),
            trajectory_ids=sumo_recording_data.trajectory_id_to_trajectory.keys(),
            ego_trajectory_id=ego_trajectory_id,
            ego_obstacle_id=-1
        )

    except Exception as e:
        logger.error(f"Failed to generate traffic for scenario {scenario_bundle.preprocessed_scenario.scenario_id}")
        logger.error(e, exc_info=True)
    finally:
        if trajectory_recorder is not None:
            trajectory_recorder.close_simulation()
    return None


def _create_trajectory_recorder(
    scenario: Scenario,
    sumo_simulation_options: SumoSimulationOptions,
) -> TrajectoryRecorder:
    sumo_simulation = SumoSimulation(initial_scenario=scenario, options=sumo_simulation_options)
    trajectory_recorder = TrajectoryRecorder(sumo_simulation=sumo_simulation)
    return trajectory_recorder
