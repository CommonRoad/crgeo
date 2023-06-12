import random
import re
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, cast

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario

from commonroad_geometric.common.io_extensions.obstacle import state_at_time


def find_scenario_files(
    base_directory: str,
    file_ext: str = '.xml',
    shuffle: bool = False,
    max_results: int = -1,
    skip_subvariants: bool = False,
    skip_scenarios: Optional[Set[str]] = None
) -> List[str]:
    """Returns a list of all CommonRoad scenario files found in a directory,
    including subdirectories.

    Args:
        base_directory (str): Base directory.
        file_ext (str): File extension to search for. Defaults to '.xml'.
        shuffle (bool): Shuffle scenarios before limiting results and filtering out subvariants.
                        If False the scenarios are sorted by file path.
        max_results (int): Maximum numbers of scenarios to return.
        skip_subvariants (bool): Filter out scenario subvariants.

    Returns:
        List[str]: List of paths to scenario files.
    """

    base_directory = str(base_directory)
    if base_directory.endswith(file_ext):
        return [base_directory]
    if skip_scenarios is not None:
        skip_scenarios = set(map(lambda f: Path(f).name, skip_scenarios))

    scenario_files = []
    scenario_numbers = []
    prefix_set = set()

    files = list(Path(base_directory).glob(f"**/*{file_ext}"))
    if shuffle:
        random.shuffle(files)
    else:
        files = sorted(files, key=lambda f: str(f))

    for file in files:
        name_prefix = re.split(r'_\d', file.name)[0]
        if not skip_subvariants or name_prefix not in prefix_set:
            # TODO: This sorting doesn't really work, should do a prior sort by scenario name
            if skip_scenarios is not None:
                if file.name in skip_scenarios:
                    continue
            scenario_files.append(str(file))
            search_result = re.search(r'-(\d+)', file.name)
            assert search_result is not None
            scenario_numbers.append(int(search_result.group(1)))
            prefix_set.add(name_prefix)
            if max_results > -1 and len(scenario_files) >= max_results:
                break

    scenario_indices = sorted(range(len(scenario_files)), key=lambda i: scenario_numbers[i])
    scenario_files_sorted = [scenario_files[i] for i in scenario_indices]
    return scenario_files_sorted


def get_dynamic_obstacles_at_timesteps(scenario: Scenario, validate: bool = False) -> Dict[int, List[DynamicObstacle]]:
    """
    Returns dictionary of dynamics obstacle at each time step.

    :param time_step: time step of interest
    :return: dictionary which maps time step to list of obstacles
    """

    timestep_dict: Dict[int, List[DynamicObstacle]] = defaultdict(list)

    for obstacle in scenario.dynamic_obstacles:
        # Full state list is [obstacle.initial_state] + obstacle.prediction.trajectory.state_list
        # Hence include initial_state.time_step for range
        for time_step in range(obstacle.initial_state.time_step, obstacle.prediction.final_time_step + 1):
            if validate:
                state_at_time(obstacle, time_step, assume_valid=True)
            timestep_dict[time_step].append(obstacle)

    return timestep_dict


class LaneletAssignmentStrategy(Enum):
    ONLY_CENTER = 'only_center'
    CENTER_FALLBACK_SHAPE = 'center_fallback_shape'
    ONLY_SHAPE = 'only_shape' # TODO: Don't assign to adjacent lanelet opposite dir?


LANELET_ASSIGNMENT_STRATEGIES_CENTER = frozenset([
    LaneletAssignmentStrategy.ONLY_CENTER,
    LaneletAssignmentStrategy.CENTER_FALLBACK_SHAPE,
])
LANELET_ASSIGNMENT_STRATEGIES_SHAPE = frozenset([
    LaneletAssignmentStrategy.ONLY_SHAPE,
    LaneletAssignmentStrategy.CENTER_FALLBACK_SHAPE,
])


def iter_dynamic_obstacles_at_timestep(scenario: Scenario, time_step: int, obstacles: Optional[List[DynamicObstacle]] = None) -> Iterable[DynamicObstacle]:
    obstacles = obstacles or scenario.dynamic_obstacles
    for obstacle in obstacles:
        if obstacle.initial_state.time_step <= time_step <= obstacle.prediction.final_time_step:
            yield obstacle


def get_dynamic_obstacles_at_timestep(scenario: Scenario, time_step: int, obstacles: Optional[List[DynamicObstacle]] = None) -> List[DynamicObstacle]:
    return list(iter_dynamic_obstacles_at_timestep(scenario=scenario, time_step=time_step, obstacles=obstacles))


def iter_unassigned_dynamic_obstacles_at_timestep(
    scenario: Scenario,
    time_step: int,
    obstacles: Optional[List[DynamicObstacle]] = None,
    check_center_lanelet_assignment: bool = True,
    check_shape_lanelet_assignment: bool = False,
) -> Iterable[DynamicObstacle]:
    assert check_center_lanelet_assignment or check_shape_lanelet_assignment
    for o in iter_dynamic_obstacles_at_timestep(scenario=scenario, time_step=time_step, obstacles=obstacles):
        center_unassigned = not check_center_lanelet_assignment or \
            o.prediction is None or \
            o.prediction.center_lanelet_assignment is None or \
            len(o.prediction.center_lanelet_assignment.get(time_step, [])) == 0
        shape_unassigned = not check_shape_lanelet_assignment or \
            o.prediction is None or \
            o.prediction.shape_lanelet_assignment is None or \
            len(o.prediction.shape_lanelet_assignment.get(time_step, [])) == 0

        if center_unassigned and shape_unassigned:
            yield o


def get_unassigned_dynamic_obstacles_at_timestep(scenario: Scenario, time_step: int, obstacles: Optional[List[DynamicObstacle]] = None) -> List[DynamicObstacle]:
    return list(iter_unassigned_dynamic_obstacles_at_timestep(scenario=scenario, time_step=time_step, obstacles=obstacles))


def get_scenario_initial_timestep(scenario: Scenario) -> int:
    if len(scenario.dynamic_obstacles) == 0:
        return 0
    return min(map(lambda o: int(o.initial_state.time_step), scenario.dynamic_obstacles))


def get_scenario_final_timestep(scenario: Scenario) -> int:
    if len(scenario.dynamic_obstacles) == 0:
        return -1
    return max(cast(int, o.prediction.final_time_step) for o in scenario.dynamic_obstacles)


def get_scenario_timestep_bounds(scenario: Scenario) -> Tuple[int, int]:
    initial_time_step = get_scenario_initial_timestep(scenario)
    final_time_step = get_scenario_final_timestep(scenario)
    return initial_time_step, final_time_step


def get_scenario_num_timesteps(scenario: Scenario) -> int:
    initial_time_step, final_time_step = get_scenario_timestep_bounds(scenario)
    return final_time_step - initial_time_step


def backup_scenario(scenario: Scenario) -> Scenario:
    copy = Scenario(
        dt=scenario.dt,
        scenario_id=scenario.scenario_id,
        author=scenario.author,
        tags=scenario.tags,
        affiliation=scenario.affiliation,
        source=scenario.source,
        location=scenario.location
    )
    copy.lanelet_network = scenario.lanelet_network
    copied_obstacles = deepcopy(scenario.dynamic_obstacles)
    copy.add_objects(copied_obstacles)
    return copy
