import argparse
import logging
import shutil
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

from commonroad_geometric.common.io_extensions.scenario_files import filter_max_scenarios
from commonroad_geometric.common.logging import setup_logging
from commonroad_geometric.dataset.scenario.generation.recording import TrajectoryMetadata
from commonroad_geometric.dataset.scenario.generation.scenario_traffic_generation import generate_traffic
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.base_scenario_preprocessor import BaseScenarioPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.filters.implementations import HighwayFilter, LaneletLengthFilter, LongestPathLengthFilter, MinIntersectionFilter, MinLaneletCountFilter, \
    TrafficFlowCycleFilter
from commonroad_geometric.dataset.scenario.preprocessing.identity_preprocessor import IdentityPreprocessor
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import CutLeafLaneletsPreprocessor, LaneletNetworkSubsetPreprocessor, MergeLaneletsPreprocessor, \
    RemoveIslandsPreprocessor, SegmentLaneletsPreprocessor
from commonroad_geometric.debugging.warnings import debug_warnings
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import SumoSimulationOptions
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.base_traffic_spawner import BaseTrafficSpawner
from commonroad_geometric.simulation.interfaces.interactive.traffic_spawning.implementations import ConstantPopulationSpawner, ConstantRateSpawner, OrnsteinUhlenbeckSpawner
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_INPUT_DIRECTORY = Path('data/t_junction_test')
# DEFAULT_SCENARIO_INPUT_DIRECTORY = Path('data/other')
DEFAULT_SCENARIO_OUTPUT_DIRECTORY = Path('outputs/generated_scenarios/')


def generate_traffic_parallel(
    input_directory: Path,
    scenario_output_directory: Path,
    num_workers: int,
    should_mute_workers: bool,
    initial_recorded_time_step: int,
    time_steps_per_run: int,
    min_trajectory_length: int,
    complete_trajectory_count: int,
    include_ego_trajectory: bool,
    time_step_cutoff: int,
    sumo_simulation_options: SumoSimulationOptions,
    metadata_log_path: Optional[Union[Path]] = None,
    max_input_scenarios: Optional[int] = None,
    preprocessor: Optional[BaseScenarioPreprocessor] = None,
    seed: Optional[int] = None,
) -> Dict[str, Optional[TrajectoryMetadata]]:
    scenario_iterator = ScenarioIterator(
        directory=Path(input_directory),
        filter_scenario_paths=partial(filter_max_scenarios, max_scenarios=max_input_scenarios),
        preprocessor=preprocessor,
        seed=seed
    )

    scenario_id_to_metadata = generate_traffic(
        input_scenario_iterator=scenario_iterator,
        sumo_simulation_options=sumo_simulation_options,
        scenario_output_dir=scenario_output_directory,
        metadata_log_path=metadata_log_path,
        num_workers=num_workers,
        should_mute_workers=should_mute_workers,
        initial_recorded_time_step=initial_recorded_time_step,
        time_steps_per_run=time_steps_per_run,
        min_trajectory_length=min_trajectory_length,
        complete_trajectory_count=complete_trajectory_count,
        # Could record for many more time-steps than number_of_time_steps_per_run
        include_ego_vehicle_trajectory=include_ego_trajectory,
        time_step_cutoff=time_step_cutoff,
    )

    return scenario_id_to_metadata


def replay_scenario(
    scenario: Path,
    renderer: TrafficSceneRenderer
) -> None:
    simulation = ScenarioSimulation(initial_scenario=scenario)
    simulation.start()
    for time_step, current_scenario in simulation:
        renderer.render(
            render_params=RenderParams(
                time_step=time_step,
                scenario=current_scenario,
                simulation=simulation,
            )
        )
    simulation.close()


def main(args) -> None:
    setup_logging(
        filename=args.log_file,
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)

    traffic_spawner: BaseTrafficSpawner
    if args.traffic_spawner == 'constant_population':
        traffic_spawner = ConstantPopulationSpawner(args.population_size)  # noqa: 405
    elif args.traffic_spawner == 'constant_rate':
        traffic_spawner = ConstantRateSpawner(
            p_spawn=args.spawn_rate, max_vehicles=args.population_size)  # noqa: 405
    else:
        traffic_spawner = OrnsteinUhlenbeckSpawner()  # noqa: 405

    sumo_simulation_options = SumoSimulationOptions(
        presimulation_steps='auto' if args.presimulation_steps is None else args.presimulation_steps,
        traffic_spawner=traffic_spawner,
        init_vehicle_speed=args.initial_vehicle_speed,
        dt=args.dt
    )

    preprocessor = IdentityPreprocessor()
    if args.subset_radius is not None:
        subset_preprocessor = LaneletNetworkSubsetPreprocessor(args.subset_radius)
        for subset in range(args.subsets_per_scenario - 1):
            subset_preprocessor |= subset_preprocessor  # Will return args.subsets_per_scenario scenarios with laneletnetworksubsets
    preprocessor >>= RemoveIslandsPreprocessor()
    if args.merge_lanelets:
        preprocessor >>= MergeLaneletsPreprocessor()
    if args.leaf_cutoff is not None:
        preprocessor >>= CutLeafLaneletsPreprocessor(args.leaf_cutoff)
    if args.segment_length is not None:
        preprocessor >>= SegmentLaneletsPreprocessor(args.segment_length)

    post_filter = LaneletLengthFilter(min_length=args.min_lanelet_length, max_length=args.max_lanelet_length)
    if args.no_highway:
        post_filter &= HighwayFilter()
    if args.min_intersections is not None:
        post_filter &= MinIntersectionFilter(min_intersections=args.min_intersections)
    if args.no_cycles:
        post_filter &= TrafficFlowCycleFilter()
    if args.size_threshold is not None:
        post_filter &= MinLaneletCountFilter(min_size_threshold=args.size_threshold)
    if args.min_longest_path is not None:
        post_filter &= LongestPathLengthFilter(min_length=args.min_longest_path)

    preprocessor >>= post_filter
    scenarios_with_traffic = generate_traffic_parallel(
        input_directory=args.scenario_dir,
        scenario_output_directory=args.scenario_output_dir,
        num_workers=args.num_workers,
        should_mute_workers=args.mute_workers,
        initial_recorded_time_step=args.initial_recorded_timestep,
        time_steps_per_run=args.timesteps_per_run,
        min_trajectory_length=args.min_trajectory_length,
        complete_trajectory_count=args.complete_trajectory_count,
        include_ego_trajectory=args.include_ego_trajectory,
        time_step_cutoff=args.timestep_cutoff,
        sumo_simulation_options=sumo_simulation_options,
        metadata_log_path=args.metadata_log_path,
        max_input_scenarios=args.max_input_scenarios,
        preprocessor=preprocessor,
        seed=42 if args.shuffle else None
    )

    _renderer = TrafficSceneRenderer()
    for (scenario_id, trajectory_recording_metadata), _ in zip(scenarios_with_traffic.items(), range(args.render_count)):
        if trajectory_recording_metadata is None:
            print(f"Failed to generate traffic for {scenario_id}")
            continue
        print(f"Replaying: {scenario_id} from {trajectory_recording_metadata}")
        replay_scenario(trajectory_recording_metadata.scenario_file_path, _renderer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate traffic for scenarios with SUMO.")
    parser.add_argument("--scenario-dir",
                        type=Path,
                        default=DEFAULT_SCENARIO_INPUT_DIRECTORY,
                        help="path to scenario directory for which traffic should be generated")
    parser.add_argument("--scenario-output-dir",
                        type=Path,
                        default=DEFAULT_SCENARIO_OUTPUT_DIRECTORY,
                        help="output directory for the scenarios with the generated traffic")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="remove and re-create the output directory before traffic generation")
    parser.add_argument("--log-file",
                        default="generate-scenario-with-traffic.log",
                        help="path to the log file")
    parser.add_argument("--metadata-log-path",
                        type=Path,
                        default=DEFAULT_SCENARIO_OUTPUT_DIRECTORY / "traffic-generation-metadata.json",
                        help="path to the json log file containing metadata about the generated traffic")
    parser.add_argument("--num-workers",
                        type=int,
                        default=1,
                        help="number of parallel workers generating traffic")
    parser.add_argument("--mute-workers",
                        action="store_true",
                        help="mute workers output to stdout")
    parser.add_argument("--initial-recorded-timestep",
                        type=int,
                        default=0,
                        help="time-step at which recording starts")
    # Specifically chosen this low as to not crash machines with only 16GB of RAM
    parser.add_argument("--timesteps-per-run",
                        type=int,
                        default=500,
                        help="how many timesteps traffic should be generated for")
    parser.add_argument("--min-trajectory-length",
                        type=int,
                        default=50,
                        help="minimum trajectory length of recorded vehicles")
    parser.add_argument("--max-input-scenarios",
                        type=int,
                        help="optional upper limit on the number of scenarios to process")
    parser.add_argument("--complete-trajectory-count",
                        type=int,
                        default=0,
                        help="how many complete trajectories should be recorded")
    parser.add_argument("--presimulation-steps",
                        type=int,
                        help="how many presimulation steps to run before recording")
    parser.add_argument("--dt",
                        type=float,
                        default=0.04,
                        help="timestep size for simulation")
    parser.add_argument("--segment-length",
                        type=float,
                        help="whether to segment the lanelets as a preprocessing step")
    parser.add_argument("--merge-lanelets",
                        action="store_true",
                        help="whether to merge the lanelets as a preprocessing step")
    parser.add_argument("--subset-radius",
                        type=float,
                        help="radius of random subset to be sampled")
    parser.add_argument("--subsets-per-scenario",
                        type=int,
                        default=1,
                        help="number of subsets to sample per scenario")
    parser.add_argument("--no-highway",
                        action="store_true",
                        help="ignore scenarios with multilane roads")
    parser.add_argument("--no-cycles",
                        action="store_true",
                        help="ignore scenarios with cycles")
    parser.add_argument("--min-intersections",
                        type=int,
                        help="only keep scenarios with at least this number of intersections")
    parser.add_argument("--include-ego-trajectory",
                        action="store_true",
                        help="whether to include an ego trajectory in the recorded traffic")
    parser.add_argument("--timestep-cutoff",
                        type=int,
                        default=30000,
                        help="timestep cutoff for traffic generation, will never record longer than this to ensure termination")
    parser.add_argument("--initial-vehicle-speed",
                        type=float,
                        default=2.5,
                        help="initial speed of spawned vehicles")
    parser.add_argument('--traffic-spawner',
                        default='constant_rate',
                        const='constant_rate',
                        nargs='?',
                        choices=['constant_population', 'constant_rate', 'ornstein_uhlenbeck'],
                        help='type of traffic spawned used for interactive SUMO simulation')
    parser.add_argument("--population-size",
                        type=int,
                        help="population size for constant_population traffic spawner")
    parser.add_argument("--spawn-rate",
                        type=float,
                        default=0.007,
                        help="spawn rate for constant_rate traffic spawner")
    parser.add_argument("--render-count",
                        type=int,
                        default=5,
                        help="how many scenarios with traffic should be rendered after recording")
    parser.add_argument("--size-threshold",
                        type=int,
                        help="minimum number of lanelets in output scenarios")
    parser.add_argument("--min-longest-path",
                        type=int,
                        help="minimum path length in resulting lanelet graph")
    parser.add_argument("--min-lanelet-length",
                        type=float,
                        help="minimum lanelet length in resulting lanelet graph")
    parser.add_argument("--max-lanelet-length",
                        type=float,
                        help="minimum lanelet length in resulting lanelet graph")
    parser.add_argument("--leaf-cutoff",
                        type=float,
                        help="cutoff exit and entrance lanelets")
    parser.add_argument("--no-warn",
                        action="store_true",
                        help="disables warnings")
    parser.add_argument("--profile",
                        action="store_true",
                        help="profiles code")
    parser.add_argument("--debug",
                        action="store_true",
                        help="activates debug logging")
    parser.add_argument("--shuffle",
                        action="store_true",
                        help="shuffle scenarios")
    args = parser.parse_args()


    def run_main() -> None:
        if args.profile:
            from commonroad_geometric.debugging.profiling import profile
            profile(main, kwargs=dict(args=args))
        else:
            main(args)


    if args.no_warn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_main()

    else:
        debug_warnings(run_main)
