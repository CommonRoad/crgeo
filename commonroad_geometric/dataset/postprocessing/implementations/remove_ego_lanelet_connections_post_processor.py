

from typing import List, Optional
import torch

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle


class RemoveEgoLaneletConnectionsPostProcessor(BaseDataPostprocessor):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:        
        
        assert ego_vehicle is not None

        # TODO: l2v??
        # TODO: update ALL features
        
        for data in samples:
            ego_obstacle_mask = torch.where(data.v.id == ego_vehicle.obstacle_id)[0]
            for ego_obstacle_idx_th in ego_obstacle_mask:
                ego_obstacle_idx = ego_obstacle_idx_th.item()
                edge_mask = data.vehicle_to_lanelet.edge_index[0, :] != ego_obstacle_idx
                data.vehicle_to_lanelet.edge_index = data.vehicle_to_lanelet.edge_index[:, edge_mask]
                data.vehicle_to_lanelet.edge_attr = data.vehicle_to_lanelet.edge_attr[edge_mask, :]
                data.vehicle_to_lanelet.arclength_rel = data.vehicle_to_lanelet.arclength_rel[edge_mask, :]
                data.vehicle_to_lanelet.arclength_abs = data.vehicle_to_lanelet.arclength_abs[edge_mask, :]
            
        return samples
