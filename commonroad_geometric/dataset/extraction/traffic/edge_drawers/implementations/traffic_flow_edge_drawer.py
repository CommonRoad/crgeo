from typing import Optional
import torch
from torch import Tensor

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations.fully_connected import FullyConnectedEdgeDrawer


class TrafficFlowEdgeDrawer(BaseEdgeDrawer):
    """
    Connects source vehicles that are either on the same lanelet as the target vehicle, 
    or one lanelets connected to the target lanelet in the traffic flow direction.
    """

    def __init__(
        self,
        base_edge_drawer: Optional[BaseEdgeDrawer] = None,
        dist_threshold: Optional[float] = None,
        hop_threshold: int = 3
    ) -> None:
        if base_edge_drawer is None:
            self.base_edge_drawer = FullyConnectedEdgeDrawer(dist_threshold=dist_threshold)
        else:
            self.base_edge_drawer = base_edge_drawer
        self.hop_threshold = hop_threshold
        super().__init__(dist_threshold)

    def _draw(self, options: BaseEdgeDrawingParams) -> Tensor:
        base_edge_index, _ = self.base_edge_drawer(options)
        simulation = options.simulation
        edges_to_keep = []
        for e in range(base_edge_index.shape[1]):
            source_obstacle_id = options.v_data['id'][base_edge_index[0][e]].item()
            target_obstacle_id = options.v_data['id'][base_edge_index[1][e]].item()

            source_lanelet_ids = simulation.obstacle_id_to_lanelet_id[source_obstacle_id]
            target_lanelet_ids = simulation.obstacle_id_to_lanelet_id[target_obstacle_id]

            relevant_source_lanelets = set()
            for target_lanelet_id in target_lanelet_ids:
                relevant_source_lanelets = set.union(
                    relevant_source_lanelets,
                    set(l for l, v in simulation.lanelet_hops[target_lanelet_id].items() if v <= self.hop_threshold)
                )

            for source_lanelet_id in source_lanelet_ids:
                if source_lanelet_id in relevant_source_lanelets:
                    edges_to_keep.append(e)
                    break

            # if target_obstacle_id == -1:
            #     print(relevant_source_lanelets)

        edge_index = base_edge_index[:, edges_to_keep]
        
        return edge_index
