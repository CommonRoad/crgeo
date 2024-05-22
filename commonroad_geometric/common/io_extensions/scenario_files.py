from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set

from commonroad_geometric.common.io_extensions.scenario_file_format import ScenarioFileFormat


def find_scenario_paths(
    directory: Path,
    file_format: ScenarioFileFormat = ScenarioFileFormat.XML,
) -> List[Path]:
    """
    Args:
        directory (str): Base directory
        file_format (ScenarioFileFormat): File extension to search for, supports: {'.xml', '.pkl'}. Defaults to: '.xml'.

    Note:
        ScenarioFileFormat.ALL will first search for '.pkl' files in the directory, then searches for '.xml' files for
        which the stem does not match any of the found '.pkl' files.

    Returns:
        List[Path]: list of paths to CommonRoad scenario files found in directory, recursively including subdirectories.

    Raises:
        ValueError, if file format is not supported
    """
    for suffix in file_format.suffixes:
        if directory.is_file() and directory.suffix == suffix:
            # Match greedily for single file
            return [directory]

    found_file_stems = set()
    files = []
    for suffix in file_format.suffixes:
        files_iter = Path(directory).glob(f"**/*{suffix}")
        for file in sorted(files_iter):
            if file.stem not in found_file_stems:
                files.append(file)
                found_file_stems.add(file.stem)
    return files


def filter_scenario_paths(
    scenario_paths: List[Path],
    excluded_scenario_names: Optional[Set[str]],
    subvariant_prefix_regex: Optional[str] = None,
    max_scenarios: Optional[int] = None,
) -> List[Path]:
    filtered_scenario_paths = scenario_paths
    if excluded_scenario_names is None:
        excluded_scenario_names = set()
    if excluded_scenario_names:
        filtered_scenario_paths = filter_scenario_filenames(filtered_scenario_paths, excluded_scenario_names)
    if subvariant_prefix_regex is not None:
        filtered_scenario_paths = filter_scenario_subvariants(filtered_scenario_paths, subvariant_prefix_regex)
    if max_scenarios is not None:
        filtered_scenario_paths = filter_max_scenarios(filtered_scenario_paths, max_scenarios)
    return filtered_scenario_paths


def filter_max_scenarios(
    scenario_paths: List[Path],
    max_scenarios: int
) -> List[Path]:
    """
    Returns at most max_scenarios paths from scenario_paths.

    Args:
        scenario_paths (List[Path]): a list of paths to scenarios
        max_scenarios (int): max amount of scenarios

    Returns:
        a new sublist of scenario_paths
    """
    return scenario_paths[:max_scenarios]


def filter_scenario_filenames(
    scenario_paths: List[Path],
    excluded_scenario_names: Set[str]
) -> List[Path]:
    """
    Returns scenario_paths for which the filename without the file extension is not in the excluded_scenario_names.

    Args:
        scenario_paths (List[Path]): a list of paths to scenarios
        excluded_scenario_names (Set[str]): set of excluded scenario names

    Returns:
        a new sublist of scenario_paths
    """
    if not excluded_scenario_names:
        return scenario_paths

    filtered_scenario_paths = []
    for scenario_path in scenario_paths:
        filename = scenario_path.stem
        if filename not in excluded_scenario_names:
            filtered_scenario_paths.append(scenario_path)
    return filtered_scenario_paths


def filter_scenario_subvariants(
    scenario_paths: List[Path],
    subvariant_prefix_regex: str = r'_\d'
) -> List[Path]:
    """
    Returns one scenario_path per set of subvariants of the same scenario as identified by their common prefix.

    Args:
        scenario_paths (List[Path]): a list of paths to scenarios
        subvariant_prefix_regex (str): Regex for the common prefix of a set of scenario subvariants

    Returns:
        a new sublist of scenario_paths
    """
    filtered_scenario_paths = []

    prefix_set = set()
    for scenario_path in scenario_paths:
        name_prefix = re.split(subvariant_prefix_regex, scenario_path.stem)[0]
        if name_prefix not in prefix_set:
            prefix_set.add(name_prefix)
            filtered_scenario_paths.append(scenario_path)

    return filtered_scenario_paths
