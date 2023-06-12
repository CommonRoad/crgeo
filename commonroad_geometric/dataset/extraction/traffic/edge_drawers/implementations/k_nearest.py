from typing import Optional

import torch
from torch import Tensor

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams


class KNearestEdgeDrawer(BaseEdgeDrawer):
    """
    Draws edges from the k nearest neighboring vehicles.
    """
    def __init__(self, k: int = 3, dist_threshold: Optional[float] = 50.0) -> None:
        super().__init__(dist_threshold=dist_threshold)
        self._k = int(k)

    def _draw(self, options: BaseEdgeDrawingParams) -> Tensor:
        assert options.dist_matrix is not None
        assert options.n_vehicles is not None
        k_nearest_source = options.dist_matrix.argsort()[:, 1:1+self._k]
        k_nearest_target = torch.arange(options.n_vehicles).unsqueeze(-1).repeat(1, k_nearest_source.shape[1])
        edge_index = torch.vstack([k_nearest_source.flatten(), k_nearest_target.flatten()])

        return edge_index
