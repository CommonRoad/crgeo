from typing import Any, List, Optional, Tuple
import torch

from crgeo.common.geometry.helpers import cut_polyline
from crgeo.dataset.commonroad_data import CommonRoadData
from crgeo.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from crgeo.simulation.base_simulation import BaseSimulation
from crgeo.simulation.ego_simulation.ego_vehicle import EgoVehicle


class VirtualEgoLaneletPostProcessor(BaseDataPostprocessor):
    """
    Inserts a virtual lanelet node at the current ego vehicle position. 
    """
    def __init__(
        self,
        lanelet_length: float
    ) -> None:
        """
        Args:
            lanelet_length (float): Length of ego-centered virtual lnaelet
        """
        self.lanelet_length = lanelet_length
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

        lanelet_id_route = ego_vehicle.ego_route.lanelet_id_route
        current_lanelet_id = next(lid for lid in simulation.obstacle_id_to_lanelet_id[ego_vehicle.obstacle_id] if lid in lanelet_id_route)
        current_lanelet_path = simulation.get_lanelet_center_polyline(current_lanelet_id)
        current_arclength = current_lanelet_path.get_projected_arclength(ego_vehicle.state.position)

        cumulative_arclength: float = 0.0
        virtual_route_buffer: List[Tuple[Any, ...]] = []
        cutoffs = []
        deletions = []
        route_counter = lanelet_id_route.index(current_lanelet_id)

        while cumulative_arclength <= self.lanelet_length:
            remaining_lanelet_distance = current_lanelet_path.length - current_arclength
            distance_to_go = self.lanelet_length - cumulative_arclength
            current_lanelet_idx = simulation.lanelet_id_to_lanelet_idx[current_lanelet_id]

            if distance_to_go < remaining_lanelet_distance:
                delta = distance_to_go
                done = True
            else:
                delta = remaining_lanelet_distance
                done = False

            lanelet_signature = (current_lanelet_idx, current_lanelet_id, current_arclength, current_arclength + delta)
            if len(virtual_route_buffer) >= 1 and not done:
                deletions.append(lanelet_signature)
            else:
                cutoffs.append(lanelet_signature)

            cumulative_arclength += delta
            virtual_route_buffer.append(lanelet_signature)

            if done:
                break
            elif route_counter == len(lanelet_id_route) - 1:
                break
            else:
                route_counter += 1
                current_lanelet_id = lanelet_id_route[route_counter]
                current_lanelet_path = simulation.get_lanelet_center_polyline(current_lanelet_id)
                current_arclength = 0.0
        
        virtual_lanelet_idx = data.lanelet.num_nodes
        virtual_lanelet_id = -1
        has_end_lanelet = len(virtual_route_buffer) > 1

        start_lanelet_idx = virtual_route_buffer[0][0]
        start_lanelet_id = virtual_route_buffer[0][1]
        start_arclength = virtual_route_buffer[0][2]
        start_lanelet_path = simulation.get_lanelet_center_polyline(start_lanelet_id)
        start_pos = torch.from_numpy(start_lanelet_path(start_arclength)).unsqueeze(0)

        end_lanelet_idx = virtual_route_buffer[-1][0]
        end_lanelet_id = virtual_route_buffer[-1][1]
        end_lanelet_path = simulation.get_lanelet_center_polyline(end_lanelet_id)
        end_arclength = virtual_route_buffer[-1][3]
        end_pos = torch.from_numpy(end_lanelet_path(end_arclength)).unsqueeze(0)

        left_vertices = data.l.left_vertices.view(data.l.num_nodes, -1, 2)
        center_vertices = data.l.center_vertices.view(data.l.num_nodes, -1, 2)
        right_vertices = data.l.right_vertices.view(data.l.num_nodes, -1, 2)
        mask_vertices = ((left_vertices != 0) | (center_vertices != 0) | (right_vertices != 0)).all(dim=2)

        start_left_vertices = left_vertices[start_lanelet_idx][mask_vertices[start_lanelet_idx]]
        start_center_vertices = center_vertices[start_lanelet_idx][mask_vertices[start_lanelet_idx]]
        start_right_vertices = right_vertices[start_lanelet_idx][mask_vertices[start_lanelet_idx]]
        start_left_polyline_cut = cut_polyline(start_left_vertices.numpy(), [start_arclength])
        start_center_polyline_cut = cut_polyline(start_center_vertices.numpy(), [start_arclength])
        start_right_polyline_cut = cut_polyline(start_right_vertices.numpy(), [start_arclength])

        if has_end_lanelet:
            end_left_vertices = left_vertices[end_lanelet_idx][mask_vertices[end_lanelet_idx]]
            end_center_vertices = center_vertices[end_lanelet_idx][mask_vertices[end_lanelet_idx]]
            end_right_vertices = right_vertices[end_lanelet_idx][mask_vertices[end_lanelet_idx]]
            end_left_polyline_cut = cut_polyline(end_left_vertices.numpy(), [end_arclength])
            end_center_polyline_cut = cut_polyline(end_center_vertices.numpy(), [end_arclength])
            end_right_polyline_cut = cut_polyline(end_right_vertices.numpy(), [end_arclength])

        route_idx_array = torch.tensor([elm[0] for elm in virtual_route_buffer], dtype=torch.long)
        ignore_keys = {'length', 'id', 'end_pos', 'start_pos', 'center_pos', 'x', 'left_vertices', 'center_vertices', 'right_vertices'}


        data.lanelet.lanelet_length = torch.cat([data.lanelet.lanelet_length, data.lanelet.lanelet_length[route_idx_array].sum(0, keepdim=True)], 0)
        data.lanelet.id = torch.cat([data.lanelet.id, torch.tensor([virtual_lanelet_id]).unsqueeze(0)], 0)
        data.lanelet.start_pos = torch.cat([data.lanelet.start_pos, start_pos], 0)
        data.lanelet.center_pos = torch.cat([data.lanelet.center_pos, start_pos], 0)
        data.lanelet.end_pos = torch.cat([data.lanelet.end_pos, end_pos], 0)
        data.lanelet.end_pos[start_lanelet_idx] = start_pos
        data.lanelet.lanelet_length[start_lanelet_idx] = start_arclength
        if has_end_lanelet:
            data.lanelet.start_pos[end_lanelet_idx] = end_pos
            data.lanelet.lanelet_length[end_lanelet_idx] -= end_arclength

        padding_size = data.l.left_vertices.shape[1]

        delete_start_lanelet = False
        if len(start_left_polyline_cut) == 2 and len(start_center_polyline_cut) == 2 and len(start_right_polyline_cut) == 2:
            start_lanelet_left_vertices_new = torch.from_numpy(start_left_polyline_cut[0].flatten())
            start_lanelet_left_vertices_new = torch.nn.functional.pad(start_lanelet_left_vertices_new, (0, padding_size - start_lanelet_left_vertices_new.shape[0]))
            start_lanelet_center_vertices_new = torch.from_numpy(start_center_polyline_cut[0].flatten())
            start_lanelet_center_vertices_new = torch.nn.functional.pad(start_lanelet_center_vertices_new, (0, padding_size - start_lanelet_center_vertices_new.shape[0]))
            start_lanelet_right_vertices_new = torch.from_numpy(start_right_polyline_cut[0].flatten())
            start_lanelet_right_vertices_new = torch.nn.functional.pad(start_lanelet_right_vertices_new, (0, padding_size - start_lanelet_right_vertices_new.shape[0]))
            data.lanelet.left_vertices[start_lanelet_idx] = start_lanelet_left_vertices_new
            data.lanelet.center_vertices[start_lanelet_idx] = start_lanelet_center_vertices_new
            data.lanelet.right_vertices[start_lanelet_idx] = start_lanelet_right_vertices_new
        else:
            delete_start_lanelet = True
        
        if has_end_lanelet:
            delete_end_lanelet = False
            if len(end_left_polyline_cut) == 2 and len(end_center_polyline_cut) == 2 and len(end_right_polyline_cut) == 2:
                end_lanelet_left_vertices_new = torch.from_numpy(end_left_polyline_cut[1].flatten())
                end_lanelet_left_vertices_new = torch.nn.functional.pad(end_lanelet_left_vertices_new, (0, padding_size - end_lanelet_left_vertices_new.shape[0]))
                end_lanelet_center_vertices_new = torch.from_numpy(end_center_polyline_cut[1].flatten())
                end_lanelet_center_vertices_new = torch.nn.functional.pad(end_lanelet_center_vertices_new, (0, padding_size - end_lanelet_center_vertices_new.shape[0]))
                end_lanelet_right_vertices_new = torch.from_numpy(end_right_polyline_cut[1].flatten())
                end_lanelet_right_vertices_new = torch.nn.functional.pad(end_lanelet_right_vertices_new, (0, padding_size - end_lanelet_right_vertices_new.shape[0]))
                data.lanelet.left_vertices[end_lanelet_idx] = end_lanelet_left_vertices_new
                data.lanelet.center_vertices[end_lanelet_idx] = end_lanelet_center_vertices_new
                data.lanelet.right_vertices[end_lanelet_idx] = end_lanelet_right_vertices_new
            else:
                delete_end_lanelet = True

        if len(start_left_polyline_cut) == 2 and len(start_center_polyline_cut) == 2 and len(start_right_polyline_cut) == 2:
            virtual_lanelet_left_vertices_new = torch.from_numpy(start_left_polyline_cut[1])
            virtual_lanelet_center_vertices_new = torch.from_numpy(start_center_polyline_cut[1])
            virtual_lanelet_right_vertices_new = torch.from_numpy(start_right_polyline_cut[1])
        elif start_arclength == 0.0:
            delete_start_lanelet = True
            virtual_lanelet_left_vertices_new = torch.from_numpy(start_left_polyline_cut[0])
            virtual_lanelet_center_vertices_new = torch.from_numpy(start_center_polyline_cut[0])
            virtual_lanelet_right_vertices_new = torch.from_numpy(start_right_polyline_cut[0])
        else:
            virtual_lanelet_left_vertices_new = torch.empty((0))
            virtual_lanelet_center_vertices_new = torch.empty((0))
            virtual_lanelet_right_vertices_new = torch.empty((0))
        for elm in virtual_route_buffer[1:-1]:
            virtual_lanelet_left_vertices_new = torch.cat([virtual_lanelet_left_vertices_new,left_vertices[elm[0], mask_vertices[elm[0]]]], 0)
            virtual_lanelet_center_vertices_new = torch.cat([virtual_lanelet_center_vertices_new, center_vertices[elm[0], mask_vertices[elm[0]]]], 0)
            virtual_lanelet_right_vertices_new = torch.cat([virtual_lanelet_right_vertices_new, right_vertices[elm[0], mask_vertices[elm[0]]]], 0)
        if has_end_lanelet:
            virtual_lanelet_left_vertices_new = torch.cat([virtual_lanelet_left_vertices_new, torch.from_numpy(end_left_polyline_cut[0])], 0)
            virtual_lanelet_center_vertices_new = torch.cat([virtual_lanelet_center_vertices_new, torch.from_numpy(end_center_polyline_cut[0])], 0)
            virtual_lanelet_right_vertices_new = torch.cat([virtual_lanelet_right_vertices_new, torch.from_numpy(end_right_polyline_cut[0])], 0)

        assert not (virtual_lanelet_left_vertices_new == 0.0).all(dim=1).any().item()
        assert not (virtual_lanelet_center_vertices_new == 0.0).all(dim=1).any().item()
        assert not (virtual_lanelet_right_vertices_new == 0.0).all(dim=1).any().item()

        virtual_lanelet_left_vertices_new = virtual_lanelet_left_vertices_new.flatten().unsqueeze(0)
        virtual_lanelet_center_vertices_new = virtual_lanelet_center_vertices_new.flatten().unsqueeze(0)
        virtual_lanelet_right_vertices_new = virtual_lanelet_right_vertices_new.flatten().unsqueeze(0)

        # transfering v2l edges to virtual lanelet
        is_ego_edge_mask = data.v.is_ego_mask.squeeze(1).gather(0, data.v2l.edge_index[0, :])
        start_lanelet_reassign_mask = (data.v2l.edge_index[1, :] == start_lanelet_idx) & ((data.v2l.arclength_abs.squeeze(1) >= start_arclength) | is_ego_edge_mask)
        data.v2l.edge_index[1, start_lanelet_reassign_mask] = virtual_lanelet_idx
        data.l2v.edge_index[0, start_lanelet_reassign_mask] = virtual_lanelet_idx
        for elm in virtual_route_buffer[1:-1]:
            lanelet_reassign_mask = data.v2l.edge_index[1, :] == elm[0]
            data.v2l.edge_index[1, lanelet_reassign_mask] = virtual_lanelet_idx
            data.l2v.edge_index[0, lanelet_reassign_mask] = virtual_lanelet_idx
        if has_end_lanelet:
            end_lanelet_reassign_mask = (data.v2l.edge_index[1, :] == end_lanelet_idx) & (data.v2l.arclength_abs.squeeze(1) <= end_arclength)
            data.v2l.edge_index[1, end_lanelet_reassign_mask] = virtual_lanelet_idx
            data.l2v.edge_index[0, end_lanelet_reassign_mask] = virtual_lanelet_idx

        data.lanelet.num_nodes += 1

        data.lanelet.left_vertices = torch.cat([data.lanelet.left_vertices, torch.nn.functional.pad(virtual_lanelet_left_vertices_new, (0, padding_size - virtual_lanelet_left_vertices_new.shape[1]))], 0)
        data.lanelet.center_vertices = torch.cat([data.lanelet.center_vertices, torch.nn.functional.pad(virtual_lanelet_center_vertices_new, (0, padding_size - virtual_lanelet_center_vertices_new.shape[1]))], 0)
        data.lanelet.right_vertices = torch.cat([data.lanelet.right_vertices, torch.nn.functional.pad(virtual_lanelet_right_vertices_new, (0, padding_size - virtual_lanelet_right_vertices_new.shape[1]))], 0)
        data.lanelet.x = torch.cat([data.lanelet.x, data.lanelet.x[(0,), :]], 0)
        
        for key in data.lanelet.keys():
            if key in ignore_keys:
                continue
            if not isinstance(data.lanelet[key], torch.Tensor):
                continue
            data.lanelet[key] = torch.cat([data.lanelet[key], data.lanelet[key][(start_lanelet_idx, ), :]], 0)

        if delete_start_lanelet:
            data.lanelet.left_vertices[start_lanelet_idx] *= 0
            data.lanelet.center_vertices[start_lanelet_idx] *= 0
            data.lanelet.right_vertices[start_lanelet_idx] *= 0

        if has_end_lanelet and delete_end_lanelet:
            data.lanelet.left_vertices[end_lanelet_idx] *= 0
            data.lanelet.center_vertices[end_lanelet_idx] *= 0
            data.lanelet.right_vertices[end_lanelet_idx] *= 0

        assert data.lanelet.left_vertices.shape[0] == data.lanelet.num_nodes
        assert data.lanelet.center_vertices.shape[0] == data.lanelet.num_nodes
        assert data.lanelet.right_vertices.shape[0] == data.lanelet.num_nodes

        return [data]

