from typing import Optional

import torch
from scipy.spatial import Voronoi
from scipy.spatial._qhull import QhullError
from shapely.geometry import LineString, Point, Polygon
from torch import Tensor

from commonroad_geometric.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams


class VoronoiEdgeDrawer(BaseEdgeDrawer):
    """
    Draws vehicle-to-vehicle edges according to a Delaunay triangulation
    of the position matrix: https://en.wikipedia.org/wiki/Delaunay_triangulation
    """

    def __init__(
        self,
        dist_threshold: Optional[float] = None,
        drop_undesired_edges: bool = False
    ):
        super().__init__(dist_threshold=dist_threshold)  # Not used
        self.drop_undesired_edges = drop_undesired_edges

    def _draw(self, options: BaseEdgeDrawingParams) -> Tensor:
        if options.n_vehicles == 2:
            # Insert edges between the two vehicles
            edge_index = torch.tensor([
                [0, 1],
                [1, 0],
            ], dtype=torch.long)
            return edge_index

        # Computing Voronoi diagram only makes sense for n_vehicles > 2
        try: 
            voronoi_diagram = Voronoi(options.pos)
        except QhullError:
            indices = torch.arange(options.n_vehicles, dtype=torch.long)
            edge_index = torch.cat([
                torch.combinations(indices).T,
                torch.combinations(indices.flip(dims=(0,))).T
            ], dim=1)
            return edge_index

        voronoi_edges = torch.from_numpy(voronoi_diagram.ridge_points)
        if self.drop_undesired_edges:
            # If you drop these two lines you will get the previous version
            keep_edge_indexes = self.keep_edges(voronoi_diagram, voronoi_edges)
            voronoi_edges = voronoi_edges[keep_edge_indexes]

        # Converting voronoi edges into bidirection graph
        edge_indexes = torch.cat([voronoi_edges, torch.flip(voronoi_edges, [1])]).type(torch.long)
        edge_indexes = edge_indexes.T
        return edge_indexes

    def line_irrelevant_polygon(
        self,
        line_coords: list[tuple[float, float]],
        polygon_coords: list[torch.Tensor]
    ) -> bool:
        """
        This function checks whether the line touches the polygon.
        """
        line = LineString(line_coords)
        polygon = Polygon(polygon_coords)

        if not line.intersects(polygon):
            return True
        else:
            return False

    def line_divides_polygon(
        self,
        line_coords: list[tuple[float, float]],
        polygon_coords: list[torch.Tensor]
    ) -> bool:
        """
        This function determines if a line divides a polygon into two sections.
        In other words, it checks if the line passes through the interior of the
        polygon while its endpoints lie outside the polygon.
        """
        line = LineString(line_coords)
        polygon = Polygon(polygon_coords)

        if line.intersects(polygon) and not (
            polygon.contains(Point(line_coords[0])) or polygon.contains(Point(line_coords[1]))):
            return True
        else:
            return False

    def keep_edges(self, voronoi_diagram: Voronoi, edge_indexes: torch.Tensor) -> list:
        """
        Decides which edges are going to be kept from voronoi diagram and
        using some conditions it returns the indexes of the edges that are
        believed to be valid edges.
        """

        polygons = []
        keep_edges = []

        # Gather all voronoi polygons
        for region in voronoi_diagram.regions:
            if not -1 in region and len(region) > 0:
                polygon = [voronoi_diagram.vertices[i] for i in region]
                polygons.append(polygon)

        # Loop through edges
        for index in range(len(edge_indexes)):
            current_edge = edge_indexes[index]
            # Get actual coordinates of this edge
            x = [voronoi_diagram.points[current_edge[0]][0], voronoi_diagram.points[current_edge[1]][0]]
            y = [voronoi_diagram.points[current_edge[0]][1], voronoi_diagram.points[current_edge[1]][1]]
            out = list(zip(*[x, y]))

            votes = 0
            irrelevant = 0
            for p in polygons:
                if self.line_divides_polygon(out, p):
                    votes += 1
                if self.line_irrelevant_polygon(out, p):
                    irrelevant += 1
                    votes += 1

            if not (votes == len(polygons) and votes != irrelevant):
                # For edges that are undesired for all the polygons
                # We collect their indices
                keep_edges.append(index)

        return keep_edges
