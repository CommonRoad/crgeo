import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from torch_geometric.loader import DataLoader

from commonroad_geometric.dataset.collection.road_network_dataset_collector import RoadNetworkDatasetCollector
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from commonroad_geometric.dataset.extraction.road_network.base_road_network_graph import CanonicalTransform
from commonroad_geometric.dataset.extraction.road_network.implementations import LaneletGraph
from commonroad_geometric.dataset.extraction.road_network.road_network_extractor_factory import RoadNetworkExtractorFactory, RoadNetworkExtractorOptions
from commonroad_geometric.plotting.plot_road_network_graph import plot_road_network_graph

logger = logging.getLogger(__name__)


RAW_DIR = "data/osm_recordings"
PROCESSED_DIR = "tutorials/output/graphdata"
PLOT_DIR = "tutorials/output/graphdata/plots"
SAMPLE_DIR = PLOT_DIR + "/training_samples"
PLOT_SAMPLES = False
GRAPH_CLS = LaneletGraph # Only LaneletEndpoint does not work properly due to string value node_ids


def plot_samples(
    graph: nx.DiGraph,
    output_filename: str,
    output_filetype: str = 'pdf',
    plot_kwargs_graph: Dict[str, Any] = {},
    figsize: Tuple[int, int] = (16, 12),
    output_dir: Optional[str] = None
    ) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    plot_road_network_graph(
                graph,
                ax=ax,
                **plot_kwargs_graph,
                node_size=figsize[0]*5
            )

    if output_dir is not None:
        output_path = os.path.join(output_dir, f"{output_filename}.{output_filetype}")
        fig.savefig(output_path)


def pre_transform(scenario: Scenario, planning_problem_set: PlanningProblemSet) -> Iterable[CommonRoadData]:
    extractor_factory = RoadNetworkExtractorFactory(RoadNetworkExtractorOptions(
        graph_cls=GRAPH_CLS,
        min_size=1,
        max_size=2,
        depth=2,
        include_radius=100.0,
        exclude_leaf_nodes=True,
        transform_mode=CanonicalTransform.TranslateRescale,
        plot=True,
        plot_dir=PLOT_DIR,
    ))

    collector = RoadNetworkDatasetCollector(
        extractor_factory=extractor_factory
    )

    for sample in collector.collect(scenario, max_samples=10, report_progress=False):
        yield sample


if __name__ == "__main__":
    # Remove pre-processed data from previous runs of this script
    # shutil.rmtree(PROCESSED_DIR, ignore_errors=True)

    # os.makedirs(PROCESSED_DIR, exist_ok=True)
    # os.makedirs(SAMPLE_DIR, exist_ok=True)

    # check out https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html and
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset for more details
    dataset: CommonRoadDataset[CommonRoadData, CommonRoadData] = CommonRoadDataset(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        # the pre_transform function transforms a CommonRoad scenario to any number of dataset samples. this parameter is not needed if the data is already processed.
        # pre_transform=pre_transform,

        # you can optionally parallelize the pre_transform step by using multiple worker processes
        # pre_transform_workers=4,
    )

    # instances of CommonRoadDataset provide a list-like interface
    print("Dataset size:", len(dataset))
    print("First sample from the dataset:", dataset[0])

    dataset_train, dataset_test = dataset.split(size=0.8, shuffle_before_split=False)
    print("Training dataset size:", len(dataset_train))
    print("Test dataset size:", len(dataset_test))

    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=False)
    print("Number of training batches:", len(train_loader))

    for idx, batch in enumerate(train_loader):
        if PLOT_SAMPLES:
            node_attrs = ["node_position", "source", "node_type"]
            graph = GRAPH_CLS.from_data(batch, node_attrs=node_attrs)
            plot_samples(graph=graph, output_filename=f"sample_{idx}", output_dir=SAMPLE_DIR)

        if idx in [0, 1]:
            print(batch)
