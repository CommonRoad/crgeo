import sys, os; sys.path.insert(0, os.getcwd())

from crgeo.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph
from crgeo.dataset.extraction.road_network.implementations import IntersectionGraph, LaneletEndpointGraph, LaneletGraph
from crgeo.dataset.extraction.road_network.types import LaneletEdgeType
from crgeo.dataset.iteration.scenario_iterator import ScenarioIterator
from crgeo.common.plotting.plot_road_network_graph import plot_road_network_graph
from crgeo.dataset.preprocessing.implementations import LaneletNetworkEdgeTypesFilterer, MergeLaneletsPreprocessor
#INPUT_SCENARIO = 'data/osm_crawled/'
#INPUT_SCENARIO = 'data/other/USA_Peach-1_1_T-1.xml'
#INPUT_SCENARIO = 'data/other/ZAM_Tjunction-1_50_T-1.xml'
#INPUT_SCENARIO = 'data/t_junction_recorded'
#INPUT_SCENARIO = 'data/osm_recordings/'
INPUT_SCENARIO = 'data/other'
#INPUT_SCENARIO = 'data'
#INPUT_SCENARIO = 'data/ind_sample'
#INPUT_SCENARIO = 'data/highway_test'
SHOW_PLOTS = True
RAISE_EXCEPTIONS = True
PLOT_SCENARIOS = True

output_dirs = [
    'output/toolpaper/road_network_graphs/lanelet',
    #'output/toolpaper/road_network_graphs/endpoint',
    #'output/toolpaper/road_network_graphs/intersection'
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
                # fig, ax = plot_scenario(
                #     scenario=graph.lanelet_network,
                #     lanelet_labels=False,
                #     title=False
                # )
                plot_road_network_graph(
                    graph,
                    ax=ax,
                    ignore_edge_types={LaneletEdgeType.PREDECESSOR, LaneletEdgeType.SUCCESSOR},
                    show=SHOW_PLOTS
                )

            plot_graph_wrapper()
