from typing import Optional, Tuple

import torch

from crgeo.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, \
    BaseEdgeDrawingParams


class NoEdgeDrawer(BaseEdgeDrawer):
    """
    Edge drawer that connects no vehicles and
    returns an empty vehicle graph.
    """
    def __init__(self, dist_threshold: Optional[float] = None) -> None: # TODO: Fix interface, make __init__ abstract in base class
        super().__init__(dist_threshold=None)

    def __call__(self, options: BaseEdgeDrawingParams) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert options.dist_matrix is not None # TODO
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index, options.dist_matrix

    def _draw(self, options: BaseEdgeDrawingParams) -> torch.Tensor:
        raise NotImplementedError()
