from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from sklearn.metrics.pairwise import euclidean_distances
from torch import Tensor

from commonroad_geometric.common.class_extensions.auto_repr_mixin import AutoReprMixin
from commonroad_geometric.common.class_extensions.string_resolver_mixing import StringResolverMixin


@dataclass
class BaseEdgeDrawingParams:
    v_data: Dict[str, Any]
    l_data: Dict[str, Any] 
    v2l_data: Dict[str, Any]
    pos: Tensor
    dist_matrix: Optional[Tensor] = None
    n_vehicles: Optional[int] = None


class BaseEdgeDrawer(ABC, AutoReprMixin, StringResolverMixin):
    """
    Call protocol for drawing v2v edges.
    """

    def __init__(self, dist_threshold: Optional[float] = None) -> None:
        self._dist_threshold = dist_threshold

    def __call__(self, options: BaseEdgeDrawingParams) -> Tuple[Tensor, Tensor]:
        """Draws graph edges based on vehicle and lanelet data.

        Args:
            options (BaseEdgeDrawingParams): Vehicle and lanelet data.

        Returns:
            2-tuple
                - Tensor: Edge index array, shape [2, num_edges]
                - Tensor: Symmetric vehicle-to-vehicle distance matrix, shape [num_vehicles, num_vehicles]
        """

        options.n_vehicles = options.pos.shape[0]
        
        if options.n_vehicles <= 1:
            # No edges exist
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            options.dist_matrix = torch.zeros((options.n_vehicles, options.n_vehicles))
        else:
            options.dist_matrix = torch.from_numpy(euclidean_distances(options.pos))
            if options.n_vehicles == 2:
                # Inserting edge between the two vehicles
                edge_index = torch.tensor([0, 1], dtype=torch.long).unsqueeze(1)
            else:
                edge_index = self._draw(options=options)

            # Filtering out valid edges
            if self._dist_threshold is not None:
                keep_edge_matrix = options.dist_matrix  <= self._dist_threshold
                keep_edge_mask = keep_edge_matrix[edge_index[0, :], edge_index[1, :]]
                edge_index = edge_index[:, keep_edge_mask]

        return edge_index, options.dist_matrix

    @abstractmethod
    def _draw(self, options: BaseEdgeDrawingParams) -> Tensor:
        ...
