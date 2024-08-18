import warnings
from typing import List, Optional

import torch

from commonroad_geometric.common.torch_utils.geometry import contains_any_rotated_rectangles
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle

from shapely.geometry import LineString
from scipy.interpolate import interp1d

import numpy as np
import math


def find_closest_points(tensor1, tensor2):
    """
    Return points from tensor1, which are closest to point of tensor2

    Args: 
        tensor1: size(m, 2)
        tensor2: size(n, 2)
    Return:
        values from tensor1 with shape = (n, 2) 
    """
    distances = torch.cdist(tensor1, tensor2)
    closest_indices = torch.argmin(distances, dim=0)
    closest_points = tensor1[closest_indices]
    return closest_points

def generate_inter_points(left_points, right_points, n, device=None):
    start = left_points.unsqueeze(-1)
    end = right_points.unsqueeze(-1)
    if device is not None:
        weights = torch.linspace(0, 1, n, device=device)
    else:
        weights = torch.linspace(0, 1, n)
    inter_points = start + weights * (end - start)
    return inter_points


def inter_points_check(point1, point2, inter_point):
    cross_product = (point2[0] - point1[0]) * (inter_point[1] - point1[1]) - (point2[1] - point1[1]) * (
            inter_point[0] - point1[0])
    # tolerance = 1e-6
    # is_coll = torch.abs(cross_product) < tolerance
    return cross_product


class Lanelet2DOccupancyPostProcessor(BaseDataPostprocessor):
    """
    Obtains lanelet occupancies by peeking into future data instances.
    Similar structure as in LaneletOccupancyPostProcessor, but modified in order to get longitudinal & latitudinal
    occupancy grid.
    Returns flattened 3D occupancy grids (longitudinal & latitudinal & temporal dimensions).
    """

    def __init__(
        self,
        time_horizon: int,
        latitudinal_resolution: int = 20,
        longitudinal_step: float = 0.5,
        priority_number_of_samples: int = 200,
        min_occupancy_ratio: Optional[float] = None
    ) -> None:
        """
        Args:
            time_horizon (int): How many steps into the future to include.

        """
        self._time_horizon = time_horizon
        # max_vehicle_count (int): Maximum number of vehicles per lanelet.
        # self._max_vehicle_count = max_vehicle_count
        self._longitudinal_step = longitudinal_step
        self._min_occ_ratio = min_occupancy_ratio
        self._latitudinal_resolution = latitudinal_resolution
        self.priority_number_of_samples = priority_number_of_samples
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        import time
        start = time.time()

        # TODO: Should be set as vehicle_to_lanelet attribute, not lanelet. Why?
        device = samples[0].v2v.edge_index.device

        # require all data to be from the same scenario
        assert all(sample.scenario_id == samples[0].scenario_id for sample in samples[1:])
        num_lanelets = samples[0].lanelet.num_nodes

        # TODO check if in all samples lanelets are the same, maybe in a better way
        assert sum([num_lanelets != samples[i].lanelet.num_nodes for i in range(len(samples))]) == 0, \
            "There are different amount lanelets for different samples"

        # this variable stores real shapes of grid fro each lanelet
        lanelets_grid_shapes = torch.zeros((num_lanelets, 2), device=device, dtype=int)

        # this variable stores new vertices for grid for each lanelet
        walks_grids = []

        for lanelet_i in range(num_lanelets):
            left_vertices = samples[0].lanelet.left_vertices[lanelet_i].view(-1, 2)
            right_vertices = samples[0].lanelet.right_vertices[lanelet_i].view(-1, 2)
            center_vertices = samples[0].lanelet.center_vertices[lanelet_i].view(-1, 2)

            l_line = LineString(left_vertices)
            r_line = LineString(right_vertices)

            num_of_vertices = int(left_vertices.shape[0])

            vertices_length_left = np.linspace(0, round(l_line.length, 5), num_of_vertices, endpoint=True)
            vertices_length_right = np.linspace(0, round(r_line.length, 5), num_of_vertices, endpoint=True)
            vertices_length_center = np.linspace(0, round(samples[0].lanelet.length[lanelet_i].item(), 5),
                                                 num_of_vertices, endpoint=True)

            assert vertices_length_left[-1] == round(l_line.length, 5), \
                f"vertices_length_left[-1]: {vertices_length_left[-1]}, l_line.length: {round(l_line.length, 5)} " \
                f"({l_line.length})"
            assert vertices_length_right[-1] == round(r_line.length, 5), \
                f"vertices_length_right[-1]: {vertices_length_right[-1]}, r_line.length: {round(r_line.length, 5)}"

            # TODO make sure that it will go till the last part of the road
            #  maybe we dont need this variable
            grid_length_center = np.arange(0, round(samples[0].lanelet.length[lanelet_i].item(),
                                                    5) + self._longitudinal_step, self._longitudinal_step)
            grid_length_center = grid_length_center[
                grid_length_center <= round(samples[0].lanelet.length[lanelet_i].item(), 5)
                ]

            # TODO maybe linestring could do it faster with gpu
            f_interp_center = interp1d(vertices_length_center, center_vertices.cpu(), axis=0)
            assert vertices_length_center[-1] > grid_length_center[-1]
            grid_c_vertices = torch.from_numpy(f_interp_center(grid_length_center)).to(device)

            f_interp_left = interp1d(vertices_length_left, left_vertices.cpu(), axis=0)
            f_interp_right = interp1d(vertices_length_right, right_vertices.cpu(), axis=0)

            # assert torch.linspace(0, l_line.length, grid_c_vertices.shape[0])[-1] != l_line.length
            # assert torch.linspace(0, r_line.length, grid_c_vertices.shape[0])[-1] != r_line.length

            x_left = np.linspace(0, round(l_line.length, 5), grid_c_vertices.shape[0])
            x_right = np.linspace(0, round(r_line.length, 5), grid_c_vertices.shape[0])

            assert x_left[-1] <= vertices_length_left[-1]
            assert x_right[-1] <= vertices_length_right[-1]

            grid_l_vertices = torch.from_numpy(f_interp_left(x_left)).to(device)
            grid_r_vertices = torch.from_numpy(f_interp_right(x_right)).to(device)

            x = generate_inter_points(grid_l_vertices[:, 0], grid_r_vertices[:, 0], self._latitudinal_resolution,
                                      device=device)
            y = generate_inter_points(grid_l_vertices[:, 1], grid_r_vertices[:, 1], self._latitudinal_resolution,
                                      device=device)

            walks_grids.append([x, y])
            lanelets_grid_shapes[lanelet_i, 0], lanelets_grid_shapes[lanelet_i, 1] = \
                grid_length_center.shape[0] - 1, self._latitudinal_resolution - 1

        hard_coded_number_of_vertices = 300
        assert lanelets_grid_shapes[:, 0].max().item() <= hard_coded_number_of_vertices
        lanelets_occupancies_discrete_grid = torch.zeros(
            (num_lanelets, len(samples), hard_coded_number_of_vertices, self._latitudinal_resolution - 1),
            device=device
        )

        # TODO only for visualization, delete later
        save_grids_x = torch.zeros(
            (num_lanelets, hard_coded_number_of_vertices + 1, self._latitudinal_resolution),
            device=device
        )
        save_grids_y = torch.zeros(
            (num_lanelets, hard_coded_number_of_vertices + 1, self._latitudinal_resolution),
            device=device
        )
        save_grids_shapes = torch.zeros((num_lanelets, 2), device=device, dtype=int)
        for lanelet_i in range(num_lanelets):
            save_grids_shapes[lanelet_i, 0] = walks_grids[lanelet_i][0].shape[0]
            save_grids_shapes[lanelet_i, 1] = walks_grids[lanelet_i][0].shape[1]
            save_grids_x[lanelet_i, :save_grids_shapes[lanelet_i, 0], :save_grids_shapes[lanelet_i, 1]] = \
                walks_grids[lanelet_i][0]
            save_grids_y[lanelet_i, :save_grids_shapes[lanelet_i, 0], :save_grids_shapes[lanelet_i, 1]] = \
                walks_grids[lanelet_i][1]

        for i_sample, data in enumerate(samples):
            for i_lanelet in data.vehicle_to_lanelet.edge_index[1].unique():
                i_lanelet = i_lanelet.item()
                edge_mask = data.vehicle_to_lanelet.edge_index[1] == i_lanelet
                vehicle_indices = data.vehicle_to_lanelet.edge_index[
                                  :,
                                  edge_mask,
                                  ][0]
                n_vehicles = vehicle_indices.shape[0]
                if n_vehicles == 0:
                    continue

                # vehicle position and length are NOT relative w.r.t. the lanelet length
                vehicle_len = data.vehicle.length[vehicle_indices, 0]
                vehicle_width = data.vehicle.width[vehicle_indices, 0]
                vehicle_pos = data.vehicle.pos[vehicle_indices]
                vehicle_orientation = data.vehicle.orientation[vehicle_indices, 0]

                closest_points = find_closest_points(
                    data.lanelet.center_vertices[i_lanelet].reshape(-1, 2), vehicle_pos
                )
                distances = ((closest_points - vehicle_pos) * (closest_points - vehicle_pos)).sum(dim=1)
                half_width = (
                        (data.lanelet.center_vertices[i_lanelet][:2] - data.lanelet.right_vertices[i_lanelet][:2]) *
                        (data.lanelet.center_vertices[i_lanelet][:2] - data.lanelet.right_vertices[i_lanelet][:2])
                ).sum()

                weight = torch.clamp((distances / half_width) * 4, min=1, max=(half_width*2).round())

                # left_vertices = data.lanelet.left_vertices[i_lanelet].view(-1, 2)
                # right_vertices = data.lanelet.right_vertices[i_lanelet].view(-1, 2)
                # x = generate_inter_points(left_vertices[:, 0], right_vertices[:, 0], self._latitudinal_resolution)
                # y = generate_inter_points(left_vertices[:, 1], right_vertices[:, 1], self._latitudinal_resolution)

                vertices_inside = contains_any_rotated_rectangles(x=walks_grids[i_lanelet][0],
                                                                  y=walks_grids[i_lanelet][1],
                                                                  cx=vehicle_pos[:, 0], cy=vehicle_pos[:, 1],
                                                                  height=vehicle_width,
                                                                  width=vehicle_len,
                                                                  angle=vehicle_orientation,
                                                                  weight=weight)

                # old version
                # a = vertices_inside.bool().to(device)
                # b = (torch.diff(a, dim=1, append=torch.zeros(a.shape[0], 1, device=device)) > 0) | a
                # c = (torch.diff(b, dim=0, append=torch.zeros(1, a.shape[1], device=device)) > 0) | b

                a_ = vertices_inside.to(device)
                b_ = torch.clamp(torch.diff(a_, dim=1, append=torch.zeros(a_.shape[0], 1, device=device)), min=0) + a_
                c_ = torch.clamp(torch.diff(b_, dim=0, append=torch.zeros(1, a_.shape[1], device=device)), min=0) + b_
                # assert (c_.bool() != c).sum() == 0
                # assert c_.max() == vertices_inside.max()

                c_[:-1, :-1].nonzero

                longitude = lanelets_grid_shapes[i_lanelet][0]
                lanelets_occupancies_discrete_grid[i_lanelet, i_sample, :longitude] = c_[:-1, :-1]

        print('spend: ', time.time() - start, 'secs')
        if len(samples) < self._time_horizon:
            self._time_horizon = 1
            print('Postprocessor: number of samples is less than _time_horizon, gonna use time horizon equal to 1')
        upper_idx = len(samples) - self._time_horizon + 1
        samples_with_occupancy = samples[:upper_idx]
        # TODO only for oversampling, delete later
        # max_samples = 16
        samples_to_keep = []
        samples_priority = []

        for i_sample, data in enumerate(samples_with_occupancy):
            # if len(samples_to_keep) >= max_samples:
            #     break

            occupancy_grid = lanelets_occupancies_discrete_grid[:, i_sample:i_sample + self._time_horizon]
            data.lanelet.occupancy_grid = \
                occupancy_grid.view(num_lanelets, -1)

            # TODO why we need this
            # data.lanelet.occupancy_time_horizon = self._time_horizon

            # Create a figure with subplots
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))  # Adjust the size as needed

            # Iterate over each lanelet
            for i in range(num_lanelets):
                ax = axes[i // 5][i % 5]  # Determine the position of the subplot
                grid = occupancy_grid[i, 0, :, :].numpy()  # Extract the grid data for the first time step
                im = ax.imshow(grid, cmap='viridis')  # Display the grid
                ax.set_title(f'Lanelet {i+1}')
                ax.axis('off')  # Hide axes ticks

            # Adjust layout
            fig.tight_layout()

            # Optional: Add a color bar
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

            # Show the plot
            plt.show()

            data.lanelet.save_grids_x = save_grids_x
            data.lanelet.save_grids_y = save_grids_y
            data.lanelet.save_grids_shapes = save_grids_shapes

            data.lanelet.occupancy_grid_real_shapes = lanelets_grid_shapes

            if data.lanelet.occupancy_grid.max() > 1 and \
                    (data.lanelet.occupancy_grid == data.lanelet.occupancy_grid.max()).sum() > 40:
                samples_priority.append(data)
            elif self._min_occ_ratio is None or \
                    data.lanelet.occupancy_grid.double().mean() > self._min_occ_ratio:
                samples_to_keep.append(data)

        if len(samples_priority) > self.priority_number_of_samples:
            print(f'\nSaved {samples_with_occupancy[0].scenario_id} '
                  f'number of samples priority only {self.priority_number_of_samples}')
            return samples_priority[:self.priority_number_of_samples]
        else:
            samples_to_keep_number = self.priority_number_of_samples - len(samples_priority)
            print(f'\nSaved {samples_with_occupancy[0].scenario_id} '
                  f'number of samples mixed {len(samples_priority) + len(samples_to_keep[:samples_to_keep_number])}')
            return samples_priority + samples_to_keep[:samples_to_keep_number]
