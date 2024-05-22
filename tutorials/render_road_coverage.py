import sys, os; sys.path.insert(0, os.getcwd())

import functools
from pathlib import Path

import pyglet

from commonroad_geometric.dataset.extraction.road_network.implementations import LaneletGraph, LaneletGraphConverter
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import SegmentLaneletsPreprocessor

pyglet.options['headless'] = True

if __name__ == "__main__":
    os.environ["PYGLET_HEADLESS"] = "1"
    # if this script crashes with an OpenGlException try running it with the PYGLET_HEADLESS environment variable set
    # PYGLET_HEADLESS=1 python tutorials/render_road_coverage.py

    scenario_iter = ScenarioIterator(
        directory=Path('data', 'other'),
        preprocessor=SegmentLaneletsPreprocessor(lanelet_max_segment_length=20.0),
    )
    for scenario_bundle in scenario_iter:
        print(scenario_bundle.preprocessed_scenario.scenario_id)
        lanelet_graph = LaneletGraph.from_scenario(
            scenario_bundle.preprocessed_scenario,
            graph_conversion_steps=[
                functools.partial(LaneletGraphConverter.render_road_coverage, size=35,
                                  lanelet_depth=6, lanelet_orientation_buckets=0),
            ],
        )
        lanelet_graph.plot()
