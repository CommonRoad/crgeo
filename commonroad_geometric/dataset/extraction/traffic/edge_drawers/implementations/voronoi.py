import scipy.spatial
import torch
from torch import Tensor

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams


class VoronoiEdgeDrawer(BaseEdgeDrawer):
    """
    Draws vehicle-to-vehicle edges according to a Delaunay triangulation
    of the position matrix: https://en.wikipedia.org/wiki/Delaunay_triangulation
    """
    def _draw(self, options: BaseEdgeDrawingParams) -> Tensor:
        if options.n_vehicles == 2:
            # Insert edges between the two vehicles
            edge_index = torch.tensor([
                [0, 1],
                [1, 0],
            ], dtype=torch.long)
            return edge_index

        # Computing Voronoi diagram only makes sense for n_vehicles > 2
        voronoi_diagram = scipy.spatial.Voronoi(options.pos)
        voronoi_edges = torch.from_numpy(voronoi_diagram.ridge_points)

        # Converting voronoi edges into bidirection graph
        edge_index = torch.zeros((2, 2*len(voronoi_edges)), dtype=torch.long)
        edge_index[:, :len(voronoi_edges)] = voronoi_edges.T
        edge_index[0, len(voronoi_edges):] = voronoi_edges[:, 1]
        edge_index[1, len(voronoi_edges):] = voronoi_edges[:, 0]

        return edge_index
