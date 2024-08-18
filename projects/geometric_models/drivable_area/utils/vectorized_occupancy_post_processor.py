from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import sys

import torch
from torch import Tensor
import numpy as np
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.common.torch_utils.geometry import contains_any_rotated_rectangles
from projects.geometric_models.drivable_area.models.decoder.vehicle_model import KinematicSingleTrackVehicleModel, KinematicSingleTrackVehicleStates, rotation_matrices_2d

def compute_vectorized_occupancy_temporal(temporal_data: CommonRoadDataTemporal, grid_size: int, *args, **kwargs):
    local_grid_occupancy_list, local_occupancy_flow_x_list, local_occupancy_flow_y_list = [], [], []

    cutoff_timestep = temporal_data.num_graphs // 2
    cutoff_data = temporal_data.get_example(cutoff_timestep)
    cutoff_ids = cutoff_data.v.id

    # TODO
    temporal_data.v.velocity[:, 1] = 0.0

    vectorized_vehicle_model = KinematicSingleTrackVehicleModel(
        velocity_bounds=(-13.6, 50.8),
        acceleration_bounds=(-11.5, 11.5),
        steering_angle_bound=1.006,
        steering_angle_delta_bound=0.4
    )

    view_velocity_long = 2.5*cutoff_data.v.velocity[:, (0,)] # TODO
    view_steering_angle = 0.15*torch.randn((cutoff_data.v.num_nodes, 1), device=cutoff_data.device)

    states = KinematicSingleTrackVehicleStates(
        position=cutoff_data.v.pos,
        velocity_long=view_velocity_long,
        acceleration_long=cutoff_data.v.acceleration[:, (1,)],
        steering_angle=view_steering_angle,
        orientation=cutoff_data.v.orientation,
        length_wheel_base=cutoff_data.v.length
    )

    # TODO
    temporal_data.v.view_velocity_long = torch.zeros_like(temporal_data.v.orientation)
    temporal_data.v.view_velocity_long[temporal_data.v.batch == cutoff_timestep] = view_velocity_long
    temporal_data.v.view_steering_angle = torch.zeros_like(temporal_data.v.orientation)
    temporal_data.v.view_steering_angle[temporal_data.v.batch == cutoff_timestep] = view_steering_angle

    for t in range(cutoff_timestep, temporal_data.num_graphs):
        data = temporal_data.get_example(t)

        velocity = torch.cat((states.velocity_long, torch.zeros_like(states.velocity_long)), dim=1)
        view_pos = data.v.pos.clone()
        view_orientation = data.v.orientation.clone()
        view_velocity = data.v.velocity.clone()
        for v_idx, v_id in enumerate(data.v.id):
            cutoff_vehicle_idx = (cutoff_ids == v_id).nonzero(as_tuple=True)[0]
            if len(cutoff_vehicle_idx) > 0:
                view_pos[v_idx] = states.position[cutoff_vehicle_idx]
                view_orientation[v_idx] = states.orientation[cutoff_vehicle_idx]
                view_velocity[v_idx] = velocity[cutoff_vehicle_idx]

        if t == cutoff_timestep:
            assert torch.allclose(view_pos, data.v.pos)
            assert torch.allclose(view_orientation, data.v.orientation)
            # assert torch.allclose(view_velocity, data.v.velocity)
 
        local_grid_occupancy, local_occupancy_flow_x, local_occupancy_flow_y = compute_vectorized_occupancy(
            data, 
            view_pos=view_pos,
            view_orientation=view_orientation,
            view_velocity=view_velocity,
            remove_ego_vehicle=True,
            grid_size=grid_size,
            *args, 
            **kwargs
        )
        local_grid_occupancy_list.append(local_grid_occupancy)
        local_occupancy_flow_x_list.append(local_occupancy_flow_x)
        local_occupancy_flow_y_list.append(local_occupancy_flow_y)

        states = vectorized_vehicle_model.compute_next_state(
            states=states,
            input=torch.zeros((cutoff_data.v.num_nodes, 2), device=cutoff_data.device),
            dt=0.2 # TODO dt is wrong ??? temporal_data.dt[cutoff_timestep]
        )

    pre_cutoff_count = int((temporal_data.v.batch < cutoff_timestep).sum())
    local_grid_occupancy_list.insert(0, torch.zeros((pre_cutoff_count, *local_grid_occupancy_list[0].shape[1:]), device=temporal_data.device))
    local_occupancy_flow_x_list.insert(0, torch.zeros((pre_cutoff_count, *local_occupancy_flow_x_list[0].shape[1:]), device=temporal_data.device))
    local_occupancy_flow_y_list.insert(0, torch.zeros((pre_cutoff_count, *local_occupancy_flow_y_list[0].shape[1:]), device=temporal_data.device))

    local_grid_occupancy = torch.cat(local_grid_occupancy_list, dim=0)
    local_occupancy_flow_x = torch.cat(local_occupancy_flow_x_list, dim=0)
    local_occupancy_flow_y = torch.cat(local_occupancy_flow_y_list, dim=0)

    # DEBUG RENDERING
    import sys
    if sys.gettrace() is not None and False:
        downsample_factor = 4  # Adjust this value as needed to reduce arrow density
        occupancy_flow_matrix = torch.cat([-local_occupancy_flow_y[..., None], -local_occupancy_flow_x[..., None]], dim=-1)
        import matplotlib.pyplot as plt
        for vid in temporal_data.v.id.unique()[10:]:
            if (temporal_data.v.id == vid).sum() < 20:
                continue
            for t, vidx in enumerate(torch.where(temporal_data.v.id == vid)[0]):
                if temporal_data.v.batch[vidx] < cutoff_timestep:
                    continue
                occupancy_mask = local_grid_occupancy[vidx, ...].detach().cpu().numpy()
                occupancy_mask = np.flip(np.flip(occupancy_mask, axis=0), axis=1)

                occupancy_flow = occupancy_flow_matrix[vidx, ...].detach().cpu().numpy()
                occupancy_flow = np.flip(np.flip(occupancy_flow, axis=0), axis=1)

                # Create a mesh grid for plotting the vector field and downsample it
                Y, X = np.mgrid[0:occupancy_flow.shape[0], 0:occupancy_flow.shape[1]]
                U = occupancy_flow[..., 0]
                V = occupancy_flow[..., 1]

                # Downsample the grid and vectors
                X_ds = X[::downsample_factor, ::downsample_factor]
                Y_ds = Y[::downsample_factor, ::downsample_factor]
                U_ds = U[::downsample_factor, ::downsample_factor]
                V_ds = V[::downsample_factor, ::downsample_factor]

                # Main figure with two subfigures for display
                plt.figure(figsize=(12, 6))

                # Subfigure 1: Occupancy
                plt.subplot(1, 2, 1)
                plt.title(f"Occupancy, t={t-cutoff_timestep}")
                plt.imshow(occupancy_mask)
                plt.xticks([])  # Disable x-axis ticks
                plt.yticks([])  # Disable y-axis ticks

                # Subfigure 2: Occupancy Flow
                plt.subplot(1, 2, 2)
                plt.title(f"Occupancy Flow, t={t-cutoff_timestep}")
                plt.imshow(occupancy_mask, alpha=0.3)  # Show the occupancy mask as a background
                plt.quiver(X_ds, Y_ds, U_ds, V_ds, scale=1, scale_units='xy', angles='xy')
                plt.xticks([])  # Disable x-axis ticks
                plt.yticks([])  # Disable y-axis ticks

                plt.tight_layout()
                plt.show()

                # Create and save individual subfigures for occupancy mask
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(occupancy_mask)
                ax1.set_xticks([])
                ax1.set_yticks([])
                fig1.savefig(f'occupancy_mask_{vid}_{t}.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig1)

                # Create and save individual subfigures for occupancy flow
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(occupancy_mask, alpha=0.3)
                ax2.quiver(X_ds, Y_ds, U_ds, V_ds, scale=1, scale_units='xy', angles='xy')
                ax2.set_xticks([])
                ax2.set_yticks([])
                fig2.savefig(f'occupancy_flow_{vid}_{t}.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig2)

    return local_grid_occupancy, local_occupancy_flow_x, local_occupancy_flow_y

def compute_vectorized_occupancy(
    data: CommonRoadData, 
    view_range: float = 70.0, 
    grid_size: int = 64, 
    is_polar: bool = False,
    view_pos: Optional[Tensor] = None,
    view_orientation: Optional[Tensor] = None,
    view_velocity: Optional[Tensor] = None,
    remove_ego_vehicle: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    device = data.device
    num_nodes = data.v.num_nodes
    cx = data.v.pos[:, 0] # (num_nodes,)
    cy = data.v.pos[:, 1] # (num_nodes,)

    view_x = view_pos[:, 0] if view_pos is not None else data.v.pos[:, 0]
    view_y = view_pos[:, 1] if view_pos is not None else data.v.pos[:, 1]
    view_orientation = view_orientation if view_orientation is not None else data.v.orientation 
    view_velocity = view_velocity if view_velocity is not None else data.v.velocity

    width = data.v.width # (num_nodes, 1)
    height = data.v.length # (num_nodes, 1)
    angle = data.v.orientation # (num_nodes, 1)
    velocity = data.v.velocity # (num_nodes, 2)

    if is_polar:
        # Prepare the polar grid
        radial_bins, angular_bins = grid_size, grid_size
        max_radius = view_range / 2
        radii = torch.linspace(0, max_radius, radial_bins, device=device)
        angles = torch.linspace(-np.pi, np.pi, angular_bins, device=device)
        rr, aa = torch.meshgrid(radii, angles, indexing='ij')  # rr is radius, aa is angle

        # Convert polar grid to Cartesian coordinates centered at (0, 0)
        xx = rr * torch.cos(aa)
        yy = rr * torch.sin(aa)

    else:
        # Step 1: Create normalized grid coordinates
        x = torch.linspace(-1, 1, grid_size, device=device)
        y = torch.linspace(-1, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y)  # These will have shape (grid_size, grid_size)

        # Step 2: Scale grid (without shift)
        xx = xx * (view_range / 2)
        yy = yy * (view_range / 2)


    # Prepare for batch operations
    cos_orientations = torch.cos(view_orientation)
    sin_orientations = torch.sin(view_orientation)

    # Step 3: Apply rotation about the origin (vectorized)
    rotated_xx = cos_orientations[:, None, None] * xx - sin_orientations[:, None, None] * yy
    rotated_yy = sin_orientations[:, None, None] * xx + cos_orientations[:, None, None] * yy

    # Step 4: Shift grid by view_center (vectorized)
    rotated_xx += view_x[:, None, None, None]
    rotated_yy += view_y[:, None, None, None]
 
    rotation_matrix_to_global = torch.cat((cos_orientations, sin_orientations, -sin_orientations, cos_orientations), dim=1).view(angle.shape[0], 2, 2)

    # Calculate the velocities in the global frame
    global_velocities = torch.einsum('ijk,ik->ij', rotation_matrix_to_global, velocity)

    # Prepare to calculate relative velocities in local frames of each vehicle A
    # Matrix of global velocities for all A to B comparisons
    velocity_matrix = global_velocities.repeat(angle.shape[0], 1).view(angle.shape[0], angle.shape[0], 2)

    # Relative velocities in global frame
    relative_velocity_matrix = velocity_matrix - view_velocity.unsqueeze(1)

    # Rotation matrix to transform from global frame back to each vehicle A's local frame
    rotation_matrix_to_local = torch.cat((cos_orientations, -sin_orientations, sin_orientations, cos_orientations), dim=1).view(angle.shape[0], 2, 2)

    # Apply rotation to convert relative velocities to the local frame of vehicle A
    local_relative_velocity_matrix = torch.einsum('ijk,ilk->ijl', rotation_matrix_to_local, relative_velocity_matrix)
    local_relative_velocity_matrix_x = local_relative_velocity_matrix[:, 0, :]
    local_relative_velocity_matrix_y = local_relative_velocity_matrix[:, 1, :]

    local_grid_occupancy_matrix = contains_any_rotated_rectangles(
        x=rotated_xx.flatten(start_dim=1),  # Flatten all but the first dimension
        y=rotated_yy.flatten(start_dim=1),  # Flatten all but the first dimension
        cx=cx, 
        cy=cy, 
        width=width + 0.5*view_range/grid_size, 
        height=height + 0.5*view_range/grid_size, 
        angle=angle,
        reduce=False
    ).view(num_nodes, grid_size, grid_size, num_nodes)

    # Removing ego vehicles
    if remove_ego_vehicle:
        indices = torch.arange(local_grid_occupancy_matrix.shape[0], device=device) 
        local_grid_occupancy_matrix[indices, :, :, indices] = 0

    if 'is_clone' in data.v:
        vmask = ~(data.v.is_clone == 1).squeeze(-1)
        local_grid_occupancy_matrix = local_grid_occupancy_matrix[:, :, :, vmask]
    else:
        vmask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    local_occupancy_flow_matrix_x = local_grid_occupancy_matrix * local_relative_velocity_matrix_x[:, None, None, vmask]
    local_occupancy_flow_matrix_y = local_grid_occupancy_matrix * local_relative_velocity_matrix_y[:, None, None, vmask]
    
    local_grid_occupancy = local_grid_occupancy_matrix.max(-1)[0].float()
    local_occupancy_flow_x = local_occupancy_flow_matrix_x.sum(-1)
    local_occupancy_flow_y = local_occupancy_flow_matrix_y.sum(-1)

    return local_grid_occupancy, local_occupancy_flow_x, local_occupancy_flow_y

def save_compute_occupancy_vectorized(data: Union[CommonRoadData, CommonRoadDataTemporal], image_size: int):
    compute_fn = compute_vectorized_occupancy_temporal if isinstance(data, CommonRoadDataTemporal) else compute_vectorized_occupancy

    occupancy, occupancy_flow_x, occupancy_flow_y = compute_fn(data, grid_size=image_size)
    polar_occupancy, polar_occupancy_flow_x, polar_occupancy_flow_y = compute_fn(data, grid_size=image_size, is_polar=True)

    data.v.occupancy = occupancy
    data.v.occupancy_flow = torch.stack([occupancy_flow_x, occupancy_flow_y], dim=-1)
    data.v.polar_occupancy = polar_occupancy
    data.v.polar_occupancy_flow = torch.stack([polar_occupancy_flow_x, polar_occupancy_flow_y], dim=-1)

class VectorizedOccupancyPostProcessor(BaseDataPostprocessor):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        for data in samples:      
            occupancy, occupancy_flow_x, occupancy_flow_y = compute_vectorized_occupancy(data=data)
            polar_occupancy, polar_occupancy_flow_x, polar_occupancy_flow_y = compute_vectorized_occupancy(data=data, is_polar=True)

            data.v.occupancy = occupancy
            data.v.occupancy_flow = torch.stack([occupancy_flow_x, occupancy_flow_y], dim=-1)
            data.v.polar_occupancy = polar_occupancy
            data.v.polar_occupancy_flow = torch.stack([polar_occupancy_flow_x, polar_occupancy_flow_y], dim=-1)
            
        return samples

