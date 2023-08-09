import sys, os; sys.path.insert(0, os.getcwd())

import json
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_geometric.common.io_extensions.scenario import find_scenario_files
from commonroad_geometric.dataset.extraction.road_network.base_road_network_graph import BaseRoadNetworkGraph
from commonroad_geometric.dataset.extraction.road_network.implementations import IntersectionGraph, LaneletEndpointGraph, LaneletGraph
from commonroad_geometric.dataset.extraction.road_network.types import LaneletEdgeType
from commonroad_geometric.debugging.errors import get_error_stack_summary
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph
from commonroad_geometric.plotting.plot_scenario import plot_scenario


INPUT_SCENARIO = 'data/osm_recordings/'
SHOW_PLOTS = True
RAISE_EXCEPTIONS = True
PLOT_SCENARIOS = False

output_dirs = ['tutorials/output/road_network_graphs/endpoint', 'tutorials/output/road_network_graphs/lanelet', 'tutorials/output/road_network_graphs/intersection']

configurations = [
    (LaneletGraph, True, False, False, False, False, {}, dict(failed=0, successful=0), output_dirs[1]),
    (LaneletEndpointGraph, True, False, False, False, False, dict(waypoint_density=15), dict(failed=0, successful=0), output_dirs[0]),
    (IntersectionGraph, True, False, True, False, True, {}, dict(failed=0, successful=0), output_dirs[2]),
]

scenario_files = find_scenario_files(INPUT_SCENARIO)


def populate_conversion_summary(summary: Dict[str, Any], file: str, location: str, line: int, detail: str) -> Dict[str, Any]:
    summary[file] = dict(
        location=location,
        line=line,
        detail=detail
    )
    summary["failed"] += 1
    return summary


def save_summary() -> None:
    for _, _, _, _, _, _, _, conversion_summary, output_dir in configurations:
        # Calculate the number of successful and failed scenarios
        succ_conversions = len(scenario_files) - conversion_summary["failed"]
        conversion_summary["successful"] = succ_conversions

        os.makedirs(output_dir, exist_ok=True)

        # save summary data as json
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w") as fp:
            json.dump(conversion_summary, fp, indent=2)

        print(f"Wrote summary to {summary_file}")


def plot_scenario_only(
    file_path: str,
    output_filetype: str = 'pdf',
    title: bool = False,
    plot_kwargs_scenario: Dict[str, Any] = {},
    figsize: Tuple[int, int] = (16, 12)
    ) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    scenario, _ = CommonRoadFileReader(file_path).open()
    plot_scenario(
        scenario.lanelet_network,
        title=title,
        ax=ax,
        **plot_kwargs_scenario
        )
    if output_dir is not None:
        output_path = os.path.join(output_dir, f"{scenario.scenario_id}.{output_filetype}")
        fig.savefig(output_path)


if __name__ == '__main__':
    print(f"Found {len(scenario_files)} scenario files:")
    for scenario_file in scenario_files:
        print(' - ' + scenario_file)

    for scenario_file in scenario_files:
        print(f"Processing {scenario_file}")
        for cls, show_waypoints, show_edge_weights, show_edge_angles, show_edge_labels, show_node_labels, conversion_options, conversion_summary, output_dir in configurations:
            def plot_graph_wrapper():

                print(f"Converting to {cls.__name__}")
                graph: BaseRoadNetworkGraph = cls.from_scenario_file(scenario_file, **conversion_options) # type: ignore
                if PLOT_SCENARIOS:
                    graph.plot(
                        title=True,
                        show=SHOW_PLOTS,
                        output_dir=output_dir,
                        plot_kwargs_scenario=dict(lanelet_labels=True),
                        plot_kwargs_graph=dict(
                            show_waypoints=show_waypoints,
                            show_edge_weights=show_edge_weights,
                            show_edge_angles=show_edge_angles,
                            show_edge_labels=show_edge_labels,
                            show_node_labels=show_node_labels,
                            ignore_edge_types={LaneletEdgeType.PREDECESSOR, LaneletEdgeType.SUCCESSOR}
                        ),
                        figsize=(12, 8)
                    )
                else:
                    plot_road_network_graph(
                        graph,
                        ignore_edge_types={LaneletEdgeType.PREDECESSOR, LaneletEdgeType.SUCCESSOR},
                        show=SHOW_PLOTS
                    )

            if RAISE_EXCEPTIONS:
                plot_graph_wrapper()
            else:
                try:
                    plot_graph_wrapper()
                except Exception:
                    file_location, line_number, detail = get_error_stack_summary()
                    print(f"Exception occured while processing {scenario_file}. Location: {file_location}, Line: {line_number}, Detail: {detail}")
                    conversion_summary = populate_conversion_summary(conversion_summary, scenario_file, file_location, line_number, detail)

                    # Plot scenario even if graph conversion fails
                    plot_scenario_only(scenario_file, plot_kwargs_scenario=dict(lanelet_labels=False))

    save_summary()
