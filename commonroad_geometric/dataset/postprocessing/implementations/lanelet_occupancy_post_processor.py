import warnings
from typing import List, Optional

import torch

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class LaneletOccupancyPostProcessor(BaseDataPostprocessor):
    """
    Obtains lanelet occupancies by peeking into future
    data instances. Returns flattened 2D occupancy grids (longitudinal & temporal dimensions).
    """

    def __init__(
        self,
        time_horizon: int,
        discretization_resolution: Optional[int] = None,
        max_vehicle_count: int = 20,
        min_occupancy_ratio: Optional[float] = None
    ) -> None:
        """
        Args:
            time_horizon (int): How many steps into the future to include.
            discretization_resolution (int): Number of discrete occupancy measurements per lanelet.
            max_vehicle_count (int): Maximum number of vehicles per lanelet.
        """
        self._time_horizon = time_horizon
        self._discretization_resolution = discretization_resolution
        self._max_vehicle_count = max_vehicle_count
        self._min_occupancy_ratio = min_occupancy_ratio
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:

        # TODO: Should be set as vehicle_to_lanelet attribute, not lanelet.
        device = samples[0].v2v.edge_index.device

        # require all data to be from the same scenario
        assert all(sample.scenario_id == samples[0].scenario_id for sample in samples[1:])
        n_lanelets = samples[0].lanelet.num_nodes

        if self._discretization_resolution is not None:
            scenario_occupancies_discrete = torch.zeros((
                n_lanelets,
                len(samples),
                self._discretization_resolution,
            ), dtype=torch.int32, device=device)

            # occupancy cells are represented by their center
            occupancy_cell_pos = (
                torch.arange(
                    end=self._discretization_resolution,
                    dtype=torch.float32,
                    device=device) + 0.5) / self._discretization_resolution

        scenario_occupancies_continuous = torch.zeros((
            n_lanelets,
            len(samples),
            self._max_vehicle_count,
            3,
        ), dtype=torch.float32, device=device)

        for i_sample, data in enumerate(samples):
            for i_lanelet in range(n_lanelets):
                edge_mask = data.vehicle_to_lanelet.edge_index[1] == i_lanelet
                vehicle_indices = data.vehicle_to_lanelet.edge_index[
                    :,
                    edge_mask,
                ][0]
                n_vehicles = vehicle_indices.shape[0]
                if n_vehicles == 0:
                    continue

                # vehicle position and length are relative w.r.t. the lanelet length
                vehicle_pos = data.vehicle_to_lanelet.v2l_lanelet_arclength_rel[edge_mask, 0]
                vehicle_len = data.vehicle.length[vehicle_indices, 0] / data.lanelet.length[i_lanelet, 0]

                # discrete
                if self._discretization_resolution is not None:
                    distances_discrete = torch.abs(vehicle_pos.unsqueeze(-1) - occupancy_cell_pos)
                    occupancies_discrete = (distances_discrete <= (
                        vehicle_len + 1 / (2 * self._discretization_resolution)).unsqueeze(-1))
                    occupancies_discrete, _ = occupancies_discrete.to(torch.int32).max(dim=0)
                    scenario_occupancies_discrete[i_lanelet, i_sample] = occupancies_discrete

                # continuous
                half_vehicle_len = vehicle_len / 2.0
                scenario_occupancies_continuous[i_lanelet, i_sample, :n_vehicles, 0] = vehicle_pos - half_vehicle_len
                scenario_occupancies_continuous[i_lanelet, i_sample, :n_vehicles, 1] = vehicle_pos + half_vehicle_len
                scenario_occupancies_continuous[i_lanelet, i_sample, :n_vehicles, 2] = 1  # vehicle indicator

                # Non-overlapping check (BROKEN)
                # vehicle_boundaries = scenario_occupancies_continuous[i_lanelet, i_sample, :n_vehicles, :2]
                # sorted_indices = vehicle_boundaries[:, 0].sort()[1]
                # vehicle_boundaries = vehicle_boundaries[sorted_indices].ravel()
                # if not vehicle_boundaries[:-1] <= vehicle_boundaries[1:].all():
                #     warnings.warn(f"Overlapping lanelet occupancy")

        upper_idx = len(samples) - self._time_horizon + 1
        samples_with_occupancy = samples[:upper_idx]
        samples_to_keep = []

        scenario_occupancies_continuous[:, :, :, 0] = torch.clamp(scenario_occupancies_continuous[:, :, :, 0], min=0.0)
        scenario_occupancies_continuous[:, :, :, 1] = torch.clamp(scenario_occupancies_continuous[:, :, :, 1], max=1.0)

        for i_sample, data in enumerate(samples_with_occupancy):
            if self._discretization_resolution is not None:
                # TODO
                data.lanelet.occupancy_discrete = scenario_occupancies_discrete[:,
                                                                                i_sample:i_sample + self._time_horizon].view(n_lanelets, -1)
                data.lanelet.occupancy_discretization_resolution = self._discretization_resolution
            data.lanelet.occupancy_continuous = scenario_occupancies_continuous[:,
                                                                                i_sample:i_sample + self._time_horizon].view(n_lanelets, -1)
            data.lanelet.occupancy_time_horizon = self._time_horizon
            if self._min_occupancy_ratio is None or data.lanelet.occupancy_continuous.mean().item() > self._min_occupancy_ratio:
                samples_to_keep.append(data)
            data.lanelet.occupancy_max_vehicle_count = self._max_vehicle_count

        return samples_to_keep
