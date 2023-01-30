from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from typing import TYPE_CHECKING

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.trajectory import State
from shapely.geometry.linestring import LineString


def count_crossed_intersections(
    lanelet_network: LaneletNetwork,
    trajectory_id: int,
    trajectory_id_to_trajectory: Dict[int, List[State]],
) -> int:
    intersection_incoming_lanelet_ids: Set[int] = set()
    for intersection in lanelet_network.intersections:
        for incoming_element in intersection.incomings:
            intersection_incoming_lanelet_ids.update(incoming_element.incoming_lanelets)

    trajectory = trajectory_id_to_trajectory[trajectory_id]
    trajectory_positions = [state.position for state in trajectory]

    lanelet_ids_for_position = lanelet_network.find_lanelet_by_position(trajectory_positions)

    visited_lanelet_ids = set()
    for lanelet_ids in lanelet_ids_for_position:
        for lanelet_id in lanelet_ids:
            visited_lanelet_ids.add(lanelet_id)

    crossed_intersections = intersection_incoming_lanelet_ids.intersection(visited_lanelet_ids)
    return len(crossed_intersections)


def count_possible_collisions(
    trajectory_id: int,
    trajectory_id_to_trajectory: Dict[int, List[State]],
) -> int:
    trajectory_linestring = _get_trajectory_linestring(trajectory=trajectory_id_to_trajectory[trajectory_id])
    count = 0
    for other_trajectory_id, trajectory in trajectory_id_to_trajectory.items():
        if trajectory_id == other_trajectory_id:
            continue
        trajectory = trajectory_id_to_trajectory[other_trajectory_id]
        if len(trajectory) <= 1:
            continue
        other_trajectory_linestring = _get_trajectory_linestring(trajectory)
        if not trajectory_linestring.intersection(other_trajectory_linestring).is_empty:
            count += 1
    return count


def _get_trajectory_linestring(
    trajectory: List[State],
) -> LineString:
    trajectory_positions = np.array([state.position for state in trajectory])
    trajectory_linestring = LineString(trajectory_positions)
    return trajectory_linestring


@dataclass
class TrajectoryMetadata:
    scenario_file_path: str
    trajectory_file_path: Optional[str]
    num_vehicles: int = 0
    ego_trajectory_length: int = 0
    crossed_intersections: int = 0
    max_possible_collisions: int = 0
