import sys, os; sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
from commonroad_geometric.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph
from commonroad_geometric.dataset.extraction.road_network.implementations import IntersectionGraph, LaneletEndpointGraph, LaneletGraph
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.dataset.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph
from commonroad_geometric.plotting.plot_scenario import plot_scenario
from commonroad_geometric.dataset.preprocessing.implementations import LaneletNetworkEdgeTypesFilterer, MergeLaneletsPreprocessor
#INPUT_SCENARIO = 'data/osm_crawled/'
#INPUT_SCENARIO = 'data/other/USA_Peach-1_1_T-1.xml'
#INPUT_SCENARIO = 'data/other/ZAM_Tjunction-1_50_T-1.xml'
#INPUT_SCENARIO = 'data/t_junction_recorded'
#INPUT_SCENARIO = 'data/osm_recordings/'
#INPUT_SCENARIO = 'data/other'
#INPUT_SCENARIO = 'data/other/ITA_CarpiCentro-1_3_T-1.xml'

#INPUT_SCENARIO = 'data'
#INPUT_SCENARIO = 'data/ind_sample'
INPUT_SCENARIO = 'data/highway_test'
SHOW_PLOTS = True
RAISE_EXCEPTIONS = True
PLOT_SCENARIOS = True
SCENARIO_DRAW_PARAMS = {
    'time_begin': 0,
    'lanelet': {
        'show_label': False,
        'draw_center_bound': False,
        'draw_start_and_direction': False,
        "fill_lanelet": True,
        "facecolor": "#eeeeee",
        'left_bound_color': "#c7c7c7",
        'right_bound_color': "#c7c7c7",
        'stop_line_color': "#c7c7c7",
        'draw_border_vertices': False,
        'draw_stop_line': True
    },
    'intersection': {
        'draw_intersections': False,
        'show_label': False
    }
}
FIGSIZE = (8, 6)

output_dirs = [
    'outputs/toolpaper/road_network_graphs/lanelet',
    #'outputs/toolpaper/road_network_graphs/endpoint',
    #'outputs/toolpaper/road_network_graphs/intersection'
]

configurations = [
    (LaneletGraph, True, False, False, False, False, {}, dict(failed=0, successful=0), output_dirs[0]),
    #(LaneletEndpointGraph, True, False, False, False, False, dict(waypoint_density=15), dict(failed=0, successful=0), output_dirs[1]),
    #(IntersectionGraph, True, False, True, False, True, {}, dict(failed=0, successful=0), output_dirs[2]),
]


def generate_lanelet_graph_conversion_plot():

    filters = [
        # LaneletNetworkEdgeTypesFilterer(required_edges_types={
        #     LaneletEdgeType.ADJACENT_LEFT,
        #     LaneletEdgeType.ADJACENT_RIGHT,
        #     LaneletEdgeType.CONFLICTING,
        #     LaneletEdgeType.DIVERGING,
        #     LaneletEdgeType.MERGING,
        #     LaneletEdgeType.PREDECESSOR,
        #     LaneletEdgeType.SUCCESSOR
        # })
    ]
    preprocessors = [
        MergeLaneletsPreprocessor()
    ]


    scenario_iterator = ScenarioIterator(
        INPUT_SCENARIO,
        preprocessors=preprocessors,
        prefilters=filters
    )

    print(f"Found {len(scenario_iterator)} scenario files")

    for scenario_bundle in scenario_iterator:
        print(f"Processing {scenario_bundle.input_scenario_file}")
        for cls, show_waypoints, show_edge_weights, show_edge_angles, show_edge_labels, show_node_labels, conversion_options, conversion_summary, output_dir in configurations:
            def plot_graph_wrapper():

                print(f"Converting to {cls.__name__}")
                graph: BaseRoadNetworkGraph = cls.from_scenario(scenario_bundle.input_scenario, **conversion_options) # type: ignore
                ax = None
                fig, ax = plot_scenario(
                    scenario=graph.lanelet_network,
                    title=False,
                    draw_params=SCENARIO_DRAW_PARAMS,
                    figsize=FIGSIZE
                )
                plot_road_network_graph(
                    graph,
                    ax=ax,
                    ignore_edge_types={LaneletEdgeType.PREDECESSOR, LaneletEdgeType.SUCCESSOR},
                    show=False,
                    linestyle="-",
                    draw_conflict_markers=True
                )
                # plt.tight_layout()
                if SHOW_PLOTS:
                    plt.show()

            plot_graph_wrapper()
