from typing import Any, List, Optional, Tuple

import torch
from torch.nn.functional import pad
import math
import numpy as np
from commonroad_geometric.common.geometry.helpers import cut_polyline
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from torch_geometric.utils import subgraph
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class VirtualEgoHighwayPostProcessor(BaseDataPostprocessor):
    """
    Creates virtual highway lanelets centered at the current ego vehicle position.
    """
    def __init__(
        self,
        lanelet_length: float
    ) -> None:
        """
        Args:
            length (float): Length of ego-centered virtual lanelets
        """
        self.lanelet_length = lanelet_length
        self.lanelet_length_one_dir = lanelet_length / 2
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        assert simulation is not None
        assert ego_vehicle is not None
        assert ego_vehicle.state is not None
        assert ego_vehicle.ego_route is not None
        assert ego_vehicle.obstacle_id is not None
        assert len(samples) == 1

        data = samples[0]

        #ego_edge_index = data.v2l.edge_index[0, :] == torch.where(data.v.id == -1)[0][0]
        #ego_arclength = data.v2l.arclength_abs[ego_edge_index][0].item()

        for (key, polyline_getter) in [
            ('left_vertices', simulation.get_lanelet_left_polyline),
            ('center_vertices', simulation.get_lanelet_center_polyline),
            ('right_vertices', simulation.get_lanelet_right_polyline),
        ]:
            vertices = []
            for lanelet_idx, lanelet_id in enumerate(data.l.id):
                polyline = polyline_getter(lanelet_id.item())
                projected_arclength = polyline.get_projected_arclength(ego_vehicle.state.position, linear_projection=True)
                projection_point = polyline(projected_arclength)
                projection_direction = polyline.get_direction(projected_arclength)
                projection_direction_vec = np.array([np.cos(projection_direction), np.sin(projection_direction)])
                startpoint = projection_point - projection_direction_vec*self.lanelet_length_one_dir
                endpoint = projection_point + projection_direction_vec*self.lanelet_length_one_dir
                vertices_lanelet = np.vstack([startpoint, endpoint])[None, :, :]
                vertices.append(vertices_lanelet)

            vertices = np.vstack(vertices)
            data.l[key] = vertices   

        for lanelet_idx, lanelet_id in enumerate(data.l.id):
            v2l_mask = data.v2l.edge_index[1, :] == lanelet_idx
            data.v2l.arclength_abs[v2l_mask, :] -= projected_arclength - self.lanelet_length_one_dir
            data.v2l.v2l_lanelet_arclength_abs[v2l_mask, :] = data.v2l.arclength_abs[v2l_mask, :]
            data.v2l.arclength_rel[v2l_mask, :] = data.v2l.arclength_abs[v2l_mask, :] / self.lanelet_length
            data.v2l.v2l_lanelet_arclength_rel[v2l_mask, :] = data.v2l.arclength_rel[v2l_mask, :]

        data.l.length[:, 0] = self.lanelet_length
        
        # relative_distances = torch.abs(data.v2l.arclength_abs - ego_arclength).squeeze(-1)
        # v2l_keep_mask = relative_distances < self.lanelet_length
        # v_keep_indices = data.v2l.edge_index[0, v2l_keep_mask].unique()
        # subset_dict = {
        #     'vehicle': v_keep_indices,
        #     'lanelet': torch.arange(data.l.num_nodes),
        # }
        # sub_data = data.subgraph(subset_dict)
        # sub_data.v.num_nodes = len(v_keep_indices)             

        # left_vertices = data.l.left_vertices.reshape(data.l.num_nodes, -1, 2)
        # center_vertices = data.l.center_vertices.reshape(data.l.num_nodes, -1, 2)
        # right_vertices = data.l.right_vertices.reshape(data.l.num_nodes, -1, 2)
        # n_vertices = center_vertices.shape[1]
        # n_vertices_keep = math.floor(2* n_vertices * self.lanelet_length/data.l.length[0].item())
        # lanelet_arclength = ((torch.arange(n_vertices)/(n_vertices - 1))[None, :]*data.l.length)
        # virtual_ego_mask = ((lanelet_arclength - ego_arclength).abs() < self.lanelet_length)[0, :]
        # virtual_ego_mask = virtual_ego_mask & (virtual_ego_mask.int().cumsum(dim=0) <= n_vertices_keep)
        # assert virtual_ego_mask.sum() == n_vertices_keep
        # data.l.left_vertices = left_vertices[:, virtual_ego_mask, :].flatten(start_dim=1)
        # data.l.center_vertices = center_vertices[:, virtual_ego_mask, :].flatten(start_dim=1)
        # data.l.right_vertices = right_vertices[:, virtual_ego_mask, :].flatten(start_dim=1)

        return [data]

