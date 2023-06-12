from typing import List, Optional

import torch
import logging
import numpy as np
from math import cos, sin

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


logger = logging.getLogger(__name__)


class LaneletEgoSequencePostProcessor(BaseDataPostprocessor):
    """
    Obtains lanelet occupancies by peeking into future
    data instances. Returns flattened 2D occupancy grids (longitudinal & temporal dimensions).
    """
    def __init__(
        self,
        max_distance: float = 100.0,
        max_sequence_length: int = 10,
        expand_route: bool = True,
        flatten: bool = True,
        padding: bool = False,
        raise_on_too_short: bool = True
    ) -> None:
        self.max_distance = max_distance
        self.max_sequence_length = max_sequence_length
        self.flatten = flatten
        self.expand_route = expand_route
        self.padding = padding
        self.raise_on_too_short = raise_on_too_short
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert len(samples) == 1
        assert ego_vehicle is not None
        assert ego_vehicle.ego_route is not None
        assert ego_vehicle.obstacle_id is not None
        assert simulation is not None
        data = samples[0]

        data_included_lanelet_ids_list = data.lanelet.id.squeeze(1).tolist()
        data_included_lanelet_ids = set(data_included_lanelet_ids_list)
        data_included_lanelet_id_map = {lid: idx for idx, lid in enumerate(data_included_lanelet_ids_list)}

        # require all data to be from the same scenario
        lanelet_id_route = [lid for lid in ego_vehicle.ego_route.lanelet_id_route if lid in data_included_lanelet_ids]

        if self.expand_route:
            while True:
                last_lanelet = simulation.lanelet_network.find_lanelet_by_id(lanelet_id_route[-1])
                successors = [s for s in last_lanelet.successor if s in data_included_lanelet_ids]
                if not successors:
                    break
                chosen = max(successors, key=lambda x: simulation.lanelet_network.find_lanelet_by_id(x).distance[-1])
                lanelet_id_route.append(chosen)

        #try:
        assert len(lanelet_id_route) > 0

        try:
            current_lanelet_candidates = [lid for lid in simulation.obstacle_id_to_lanelet_id[ego_vehicle.obstacle_id] if lid in lanelet_id_route]
            if not len(current_lanelet_candidates):
                logger.warning(f"Found not lanelet canditates for the ego vehicle")
                return samples
            current_lanelet_candidates_set = current_lanelet_candidates
            for candidate_id in current_lanelet_candidates:
                current_lanelet_path = simulation.get_lanelet_center_polyline(candidate_id)
                candidate_lanelet = simulation.lanelet_network.find_lanelet_by_id(candidate_id)
                if any((predecessor in current_lanelet_candidates_set for predecessor in candidate_lanelet.predecessor)):
                    continue
                
                yaw = ego_vehicle.state.orientation
                l = ego_vehicle.parameters.l
                p_rear = ego_vehicle.state.position -l/2*np.array([cos(yaw), sin(yaw)])


                initial_arclength = max(0.0, current_lanelet_path.get_projected_arclength(p_rear))
                if initial_arclength >= 0.0:
                    break
            current_lanelet_id = candidate_id    

        except KeyError as e:
            print("WARNING", repr(e))
            current_lanelet_id = lanelet_id_route[0]
        current_arclength = initial_arclength

        cumulative_arclength: float = 0.0
        route_buffer = []
        route_counter = lanelet_id_route.index(current_lanelet_id)

        while cumulative_arclength < self.max_distance:
            remaining_lanelet_distance = current_lanelet_path.length - current_arclength
            distance_to_go = self.max_distance - cumulative_arclength
            current_lanelet_idx = data_included_lanelet_id_map[current_lanelet_id]

            if distance_to_go < remaining_lanelet_distance:
                delta = distance_to_go
                done = True
            else:
                delta = remaining_lanelet_distance
                done = False

            lanelet_signature = (
                current_lanelet_idx,
                current_lanelet_id,
                cumulative_arclength / self.max_distance,
                1 - current_arclength / current_lanelet_path.length,
                (current_arclength + delta) / current_lanelet_path.length,
                current_lanelet_path.length / self.max_distance
            )

            cumulative_arclength += delta
            route_buffer.append(lanelet_signature)

            if done:
                break
            elif route_counter == len(lanelet_id_route) - 1:
                break
            else:
                route_counter += 1
                current_lanelet_id = lanelet_id_route[route_counter]
                current_lanelet_path = simulation.get_lanelet_center_polyline(current_lanelet_id)
                current_arclength = 0.0

        if self.raise_on_too_short and cumulative_arclength < self.max_distance:
            raise ValueError(f"Extracted path was too short (cumulative_arclength < {cumulative_arclength:.2f} < self.max_distance {self.max_distance:.2f})")

        walk_length = min(len(route_buffer), self.max_sequence_length)
        tensor_size = walk_length if not self.padding else self.max_sequence_length
        trajectory_sequence = torch.zeros((tensor_size, 4), dtype=torch.float32, requires_grad=False)
        sequence_mask = torch.zeros((tensor_size, 1), dtype=torch.bool, requires_grad=False)
        walks = torch.zeros((1, tensor_size), dtype=torch.long, requires_grad=False)
        walk_start_length = torch.tensor(([initial_arclength]), dtype=torch.float32, requires_grad=False)

        data.walk_start_length = walk_start_length

        for i, (lanelet_idx, lanelet_id, cumulative_arclength, current_arclength, next_arclength, current_length) in enumerate(route_buffer):
            if i >= self.max_sequence_length:
                break
            trajectory_sequence[i, :] = torch.tensor([
                cumulative_arclength, current_arclength, next_arclength, current_length
            ])
            sequence_mask[i, :] = True
            walks[:, i] = lanelet_idx

        if self.flatten:
            data.ego_trajectory_sequence = trajectory_sequence.flatten()
            data.ego_trajectory_sequence_mask = sequence_mask.flatten()
            data.walks = walks.flatten()
        else:
            data.ego_trajectory_sequence = trajectory_sequence
            data.ego_trajectory_sequence_mask = sequence_mask
            data.walks = walks

        return samples
