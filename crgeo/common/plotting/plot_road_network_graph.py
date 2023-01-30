import warnings
from typing import Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from crgeo.common.geometry.continuous_polyline import ContinuousPolyline
from crgeo.dataset.extraction.road_network.types import LaneletEdgeTypeColorMap, LaneletNodeTypeColorMap
from crgeo.dataset.extraction.road_network.types import LaneletEdgeType

warnings.filterwarnings("ignore", module=r"matplotlib\..*")


# TODO xlim, ylim invariance 

def plot_road_network_graph(
    graph: nx.DiGraph,
    has_equal_axes: bool = False,
    ax: Optional[Axes] = None,
    arrow_size: int = 12,
    draw_conflict_markers: bool = False,
    draw_nodes: bool = False,
    edge_alpha: float = 0.5,
    edge_connection_style: Optional[str] = 'arc3,rad=0.2',
    edge_type_legend: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    ignore_edge_types: Optional[Set[LaneletEdgeType]] = None,
    linewidth: float = 1.25,
    node_alpha: float = 0.7,
    node_offset_meters: float = 12.5,
    node_size: int = 40,
    show: bool = False,
    show_axes: bool = False,
    show_edge_angles: bool = False,
    show_edge_labels: bool = False,
    show_edge_weights: bool = False,
    show_node_labels: Union[bool, Sequence[int]] = False,
    show_waypoints: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
        Returns a plot of the road network graph.

    Note:
        Implicitly accesses the following node or edge attributes from nx.DiGraph:
        'node_position': (One) Position of the node of the lanelet.
        'node_waypoints': (Many) Waypoints of the lanelet.
        'node_type': The (LaneletNodeType), determines the node style.
        'edge_waypoints': Waypoints of the edge between nodes.
        'lanelet_edge_type': The (LaneletEdgeType), determines the edge style.
        'edge_cardinality': Cardinality, how many edges are represented in this edge.
        'weight': Edge weight, consistent with use in networkx, important for nx.linalg.graphmatrix.adjacency_matrix
        'start_angle': Unused
        'exit_angle': Use as edge angle if show_edge_angles is True.
        'lanelet_id': Use as edge label if show_edge_labels is True.
    Args:
        graph (nx.DiGraph): Lanelet network graph.
        has_equal_axes (bool): True if axes should have equal aspect ratio. Defaults to True.
        ax Optional[Axes]: Optional custom axis.
        show_waypoints (bool): True if plot has waypoints. Defaults to True.
        show_edge_weights (bool): True if plot has edge weights. Defaults to False.
        show_edge_angles (bool): True if plot has edge angles. Defaults to True.
        show_edge_labels (bool): True if plot has edge labels. Defaults to False.
        show_node_labels (Union[bool, Sequence[int]]): Either True if plot has node labels or a sequence of labels. Defaults to True.
        node_size (int): Size of the nodes. Defaults to 80.
        cmap (Colormap): Color map for the plot. Defaults to qualitative color map "tab20" from matplotlib.
        show (bool): True if plot should be shown. Default to False.

    Returns:
        fig (Figure) - The plot of the network graph.
    """
    
    if ax is not None:
        fig = None
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if show_axes:
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    else:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set(ylabel=None)
        ax.set(yticklabels=[])
        ax.set(xlabel=None)
        ax.set(xticklabels=[])
    ignore_edge_types = ignore_edge_types if ignore_edge_types is not None else set()

    if has_equal_axes:
        ax.axis('equal')
        ax.set_aspect('equal', 'box')

    node_start_positions = nx.get_node_attributes(graph, 'start_pos')
    node_center_positions = nx.get_node_attributes(graph, 'center_pos')
    node_start_offset_positions = {}
    node_center_offset_positions = {}
    node_end_offset_positions = {}
    
    node_waypoints = nx.get_node_attributes(graph, 'center_vertices')
    # node_waypoint_lists = [p for p in list(node_waypoints.values()) if p is not None]
    # edge_waypoints = nx.get_edge_attributes(graph, 'edge_waypoints')
    # edge_waypoint_lists = [p for p in list(edge_waypoints.values()) if p is not None]
    # waypoint_lists = node_waypoint_lists # + edge_waypoint_lists
    for node, node_waypoints in node_waypoints.items():
        polyline_draw = ContinuousPolyline(node_waypoints, waypoint_resolution=40, linestring_resolution=40)
        polyline_exact = ContinuousPolyline(node_waypoints, waypoint_resolution=200, linestring_resolution=200)
        
        waypoints = polyline_draw.waypoints
        if show_waypoints:
            line = ax.plot(
                waypoints[:, 0],
                waypoints[:, 1],
                #s=1.0,
                color=LaneletEdgeTypeColorMap[LaneletEdgeType.SUCCESSOR],
                linestyle='solid',
                linewidth=linewidth,
                alpha=edge_alpha,
                #marker='x'
            )[0]
            #add_arrow(line, color='black', position=polyline(0))
            start = polyline_exact(node_offset_meters - 0.55)
            end = polyline_exact(node_offset_meters - 0.3)
            add_arrow(
                line,
                color=(*LaneletEdgeTypeColorMap[LaneletEdgeType.SUCCESSOR], edge_alpha),
                start=start,
                end=end,
                size=arrow_size
            )

        node_start_offset_positions[node] = polyline_exact(node_offset_meters)
        node_center_offset_positions[node] = polyline_exact(polyline_exact.length/2 - node_offset_meters/2)
        node_end_offset_positions[node] = polyline_exact(polyline_exact.length - node_offset_meters)
    
    if draw_nodes:
        node_types = nx.get_node_attributes(graph, 'node_type')
        node_to_color = {node: LaneletNodeTypeColorMap[type] for node, type in node_types.items() if node in node_center_positions}
        node_color_list = [node_to_color[n] if n in node_to_color else (0.2, 0.2, 0.2, 0.01) for n in graph.nodes()]
        nx.draw_networkx_nodes(
            graph,
            node_start_offset_positions,
            node_size=node_size,
            ax=ax,
            node_color=node_color_list,
            alpha=node_alpha
        )

    if show_node_labels:
        if isinstance(show_node_labels, bool):
            draw_pos = node_start_positions
        else:
            draw_pos = {k: node_start_positions[k] for k in show_node_labels}
        labels = {k: k for k in draw_pos.keys()}
        nx.draw_networkx_labels(
            graph,
            pos=draw_pos,
            labels=labels,
            ax=ax,
            font_color='black',
            font_size=6,
            font_weight='heavy'
        )

    edge_types = nx.get_edge_attributes(graph, 'lanelet_edge_type')
    edge_types = {e: t for e, t in edge_types.items() if t not in ignore_edge_types}
    edge_types_included = list(set(edge_types.values()))
    
    if edge_type_legend:
        edge_legend = [
            mpatches.Patch(
                color=LaneletEdgeTypeColorMap[LaneletEdgeType(t)],
                label=LaneletEdgeType(t).name.lower().replace('_', ' ')
            ) for t in edge_types_included
        ]
        if LaneletEdgeType.SUCCESSOR in ignore_edge_types:
            edge_legend.insert(0, mpatches.Patch(
                color=LaneletEdgeTypeColorMap[LaneletEdgeType(LaneletEdgeType.SUCCESSOR)],
                label=LaneletEdgeType(LaneletEdgeType.SUCCESSOR).name.lower().replace('_', ' ')
            ))
        
        plt.legend(handles=edge_legend)

    edge_cardinalities = nx.get_edge_attributes(graph, 'edge_cardinality')
    if edge_cardinalities:
        raise NotImplementedError()
        nx.draw_networkx_edges(
            graph,
            pos=node_center_positions,
            edgelist=edge_types,
            ax=ax,
            edge_color=edge_colors_list,
            width=[2 * (x - 1) + 1 for x in edge_cardinalities.values()],
            style='dashed',
            alpha=alpha
        )
    else:
        edge_types_start = {e: t for e, t in edge_types.items() if t not in {LaneletEdgeType.CONFLICTING, LaneletEdgeType.MERGING}}
        edge_colors_start = [LaneletEdgeTypeColorMap[LaneletEdgeType(edge_types_start[edge])] for edge in edge_types_start]
        nx.draw_networkx_edges(
            graph,
            pos=node_start_offset_positions,
            edgelist=edge_types_start,
            ax=ax,
            edge_color=edge_colors_start,
            style='solid',
            alpha=edge_alpha,
            width=linewidth,
            min_source_margin=0,#0.1,
            min_target_margin=0,#,0.1,
            connectionstyle=edge_connection_style,
            arrowsize=arrow_size
        )

        edge_types_end = {e: t for e, t in edge_types.items() if t == LaneletEdgeType.MERGING}
        edge_colors_end = [LaneletEdgeTypeColorMap[LaneletEdgeType(edge_types_end[edge])] for edge in edge_types_end]
        nx.draw_networkx_edges(
            graph,
            pos=node_start_offset_positions,
            edgelist=edge_types_end,
            ax=ax,
            edge_color=edge_colors_end,
            style='solid',
            alpha=edge_alpha,
            width=linewidth,
            min_source_margin=0,#0.1,
            min_target_margin=0,#,0.1,
            connectionstyle=edge_connection_style,#'arc3,rad=0.0',
            arrowsize=arrow_size
        )

        edge_types_center = {e: t for e, t in edge_types.items() if t == LaneletEdgeType.CONFLICTING}
        edge_colors_center = [LaneletEdgeTypeColorMap[LaneletEdgeType(edge_types_center[edge])] for edge in edge_types_center]
        nx.draw_networkx_edges(
            graph,
            pos=node_start_offset_positions,
            edgelist=edge_types_center,
            ax=ax,
            edge_color=edge_colors_center,
            style='solid',
            alpha=edge_alpha,
            width=linewidth,
            min_source_margin=0,#0.1,
            min_target_margin=0,#,0.1,
            connectionstyle=edge_connection_style,
            arrowsize=arrow_size # 'arc3,rad=0.2'
        )

    if draw_conflict_markers:
        edge_positions = nx.get_edge_attributes(graph, 'edge_position')
        locations = []
        for edge, edge_type in edge_types.items():
            if LaneletEdgeType(edge_type) == LaneletEdgeType.CONFLICTING:
                locations.append(edge_positions[edge])
        if locations:
            locations = np.vstack(locations)
            plt.scatter(
                locations[:, 0],
                locations[:, 1],
                marker='x',
                c=(1.0,0.0,0.0,0.6),
                s=90,
                zorder=10,
                linewidths=linewidth*2
            )


    if show_edge_weights or show_edge_angles or show_edge_labels:
        edge_to_angle = nx.get_edge_attributes(graph, 'exit_angle')
        edge_to_weight = nx.get_edge_attributes(graph, 'weight')
        edge_to_label = nx.get_edge_attributes(graph, 'lanelet_id')  # Use lanelet_id as label
        edge_labels = dict()
        for edge in graph.edges():
            label_parts = []
            if show_edge_labels and edge in edge_to_label:
                label_parts.append(f"ID:{edge_to_label[edge]}")
            if show_edge_weights and edge in edge_to_weight:
                label_parts.append(f"{edge_to_weight[edge]:.1f}m")
            if show_edge_angles and edge in edge_to_angle:
                label_parts.append(f"{180 / np.pi * edge_to_angle[edge]:.0f}°")
            label = ', '.join(label_parts)
            edge_labels[edge] = label
        nx.draw_networkx_edge_labels(
            graph,
            node_center_positions,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6,
            font_weight='heavy',
        )

    # xy = np.vstack(node_start_positions.values())
    # plt.scatter(xy[:, 0], xy[:, 1], s=node_size/3, c='black', edgecolors='black', facecolor='black', zorder=10)

    xy = np.vstack(node_start_offset_positions.values())
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c='black',
        edgecolors='black',
        facecolor='black',
        zorder=10,
        alpha=node_alpha
    )

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if show:
        plt.show()

    return fig


def add_arrow(line, start, end, direction='right', size=10, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    line.axes.annotate('',
        xytext=(start[0], start[1]),
        xy=(end[0], end[1]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )