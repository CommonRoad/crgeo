import torch

from crgeo.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams


class FullyConnectedEdgeDrawer(BaseEdgeDrawer):
    """
    Draws a fully connected graph (edges between all vehicles).
    For connecting only neighboring vehicles within a certain radius, use the dist_threshold parameter.
    """
    def _draw(self, options: BaseEdgeDrawingParams) -> torch.Tensor:
        assert options.n_vehicles is not None
        indices = torch.arange(options.n_vehicles, dtype=torch.long)
        edge_index = torch.cat([
            torch.combinations(indices).T,
            torch.combinations(indices.flip(dims=(0,))).T
        ], dim=1)
        return edge_index
