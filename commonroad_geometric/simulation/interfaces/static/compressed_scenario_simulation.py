from __future__ import annotations

import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker
from commonroad_dc.pycrcc import CollisionChecker  # noqa
from pandas import DataFrame

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.common.io_extensions.scenario import backup_scenario, get_dynamic_obstacles_at_timesteps
from commonroad_geometric.simulation.exceptions import SimulationRuntimeError
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulation, ScenarioSimulationOptions

# TODO delete?

def reconstruct_scenario(
    scenario_to_be_reconstructed: Scenario,
    dynamic_obstacle_id_to_trajectory_dataframe: Dict[int, DataFrame],
    should_check_trajectory_integrity = False,
) -> Scenario:
    """

    Args:
        scenario_to_be_reconstructed (Scenario):
        dynamic_obstacle_id_to_trajectory_dataframe (Dict[int, DataFrame]):
        should_check_trajectory_integrity (bool): Whether the dynamic obstacle stubs and the trajectory dataframe start and end points should be compared with each other. Defaults to False.

    Returns:

    """
    for dynamic_obstacle in scenario_to_be_reconstructed.dynamic_obstacles:
        initial_time_step = dynamic_obstacle.initial_state.time_step
        final_time_step = dynamic_obstacle.prediction.final_time_step
        initial_state = dynamic_obstacle.initial_state
        # dynamic_obstacle.state_at_time(final_time_step) is not usable here since the state_list stub only contains 2 elements
        # state_at_time relies on the length of said state_list to index and fails with final_time_step, thus HACK:
        final_state = state_at_time(dynamic_obstacle, initial_time_step + 1, assume_valid=True)

        # Assert trajectory integrity by checking that initial and final state match with end of trajectory dataframe
        if dynamic_obstacle.obstacle_id == -1:  # ego vehicle exclusion hack
            continue
        if should_check_trajectory_integrity:
            _assert_trajectory_integrity(trajectory_dataframe=dynamic_obstacle_id_to_trajectory_dataframe[dynamic_obstacle.obstacle_id],
                                         initial_state=initial_state,
                                         final_state=final_state)

        # Load full state list necessary for ScenarioSimulation
        trajectory_dataframe = dynamic_obstacle_id_to_trajectory_dataframe[dynamic_obstacle.obstacle_id]
        dynamic_obstacle.prediction.trajectory.state_list, dynamic_obstacle.prediction.occupancy_set = _load_state_list_occupancy_set(
            dynamic_obstacle_stub=dynamic_obstacle,
            trajectory_dataframe=trajectory_dataframe,
            initial_time_step=initial_time_step,
            final_time_step=final_time_step
        )
        dynamic_obstacle.prediction.center_lanelet_assignment = {}
    return scenario_to_be_reconstructed


def _assert_trajectory_integrity(
    trajectory_dataframe: DataFrame,
    initial_state: State,
    final_state: State
) -> None:
    # See https://gitlab.lrz.de/cps/commonroad-geometric/-/issues/73
    from commonroad.common.file_writer import DecimalPrecision
    possible_error = pow(0.1, DecimalPrecision.decimals)

    for attribute in trajectory_dataframe.keys():
        initial_trajectory_attribute = trajectory_dataframe[attribute].iloc[0]
        final_trajectory_attribute = trajectory_dataframe[attribute].iloc[-1]
        # HACK due to weirdness of CommonRoadFileWriter cutting of floats during scenario XML file creation
        assert (initial_trajectory_attribute - initial_state.__getattribute__(attribute) <= possible_error).all()
        assert (final_trajectory_attribute - final_state.__getattribute__(attribute) <= possible_error).all()


def _load_state_list_occupancy_set(
    dynamic_obstacle_stub: DynamicObstacle,
    trajectory_dataframe: DataFrame,
    initial_time_step: int,
    final_time_step: int,
) -> Tuple[List[State], List[Occupancy]]:
    state_list = []
    occcupancy_set = []
    for time_step in range(initial_time_step, final_time_step + 1):
        trajectory_index = time_step - initial_time_step
        kwargs = {attribute: trajectory_dataframe[attribute][trajectory_index] for attribute in trajectory_dataframe.keys()}
        state = State(time_step=time_step, **kwargs)

        shape = Rectangle(width=dynamic_obstacle_stub.obstacle_shape.width,
                          length=dynamic_obstacle_stub.obstacle_shape.length,
                          center=state.position,
                          orientation=state.orientation)

        occupancy = Occupancy(time_step=time_step, shape=shape)
        state_list.append(state)
        occcupancy_set.append(occupancy)
    return state_list, occcupancy_set


@dataclass
class CompressedSimulationOptions(ScenarioSimulationOptions):
    """
        trajectory_pickle_file (Optional[Union[str, Path]]): The .pkl file path to the dictionary with trajectory information.
    """
    trajectory_pickle_file: Optional[Union[str, Path]] = None


class CompressedScenarioSimulation(ScenarioSimulation):
    def __init__(
        self,
        initial_scenario: Union[Scenario, str],
        options: CompressedSimulationOptions,
    ) -> None:
        """
        Class for simulations based on CommonRoad scenarios with externally saved trajectory information for on-disk space savings ("compression").
        Uses pickled dynamic_obstacle_id_to_trajectory for trajectory information of dynamic obstacle stubs in initial_scenario.
        Scenario and pickle-file of trajectories can be created by TrajectoryRecorder.save_scenario_pickle_trajectories
        Uses TrafficSceneRenderer for rendering.

        Args:
            initial_scenario (Union[Scenario, str]): Initial scenario for this simulation, only has to contain dynamic obstacle stubs, remaining trajectory data loaded from pickle.
            options (BaseSimulationOptions): Options for this simulation.
        """
        trajectory_pickle_file = None
        if options.trajectory_pickle_file is not None:
            trajectory_pickle_file = options.trajectory_pickle_file
        if isinstance(initial_scenario, str):
            if trajectory_pickle_file is None and initial_scenario.endswith('.xml'):
                # This is just trying to make a lucky guess if only the scenario file was passed
                trajectory_pickle_file = initial_scenario.replace('.xml', '.pkl', 1)
            initial_scenario, _ = CommonRoadFileReader(filename=initial_scenario).open()
        if trajectory_pickle_file is None:
            raise SimulationRuntimeError(f"Missing pickled trajectory file for scenario {initial_scenario}")

        try:
            with open(trajectory_pickle_file, 'rb') as pickled_trajectories:
                dynamic_obstacle_id_to_trajectory_dataframe: Dict[int, DataFrame] = pickle.load(pickled_trajectories)
        except FileNotFoundError as e:
            raise SimulationRuntimeError(f"Could not load pickled trajectory from file: {trajectory_pickle_file}") from e

        reconstructed_scenario = reconstruct_scenario(scenario_to_be_reconstructed=initial_scenario,
                                                      dynamic_obstacle_id_to_trajectory_dataframe=dynamic_obstacle_id_to_trajectory_dataframe)
        super().__init__(
            initial_scenario=reconstructed_scenario,
            options=options,
        )
        self._options = options  # for correct type annotation

    def _reset(self) -> None:
        self._current_collision_checker = None
        self._current_scenario = backup_scenario(self.initial_scenario)
        if self._options.remove_ego_vehicle_from_obstacles:
            self.despawn_ego_vehicle(ego_vehicle=None)
        if self._options.collision_checking:
            self._current_collision_checker_time_step = -1
            self._time_variant_collision_checker: CollisionChecker = create_collision_checker(self.current_scenario)
            self._time_step_to_obstacles = get_dynamic_obstacles_at_timesteps(self.current_scenario)
