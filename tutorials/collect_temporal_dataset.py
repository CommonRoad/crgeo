import sys; import os; sys.path.insert(0, os.getcwd())

from pathlib import Path

from commonroad_geometric.dataset.collection.temporal_dataset_collector import TemporalDatasetCollector
from commonroad_geometric.dataset.commonroad_data_temporal import CommonRoadDataTemporal
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset, CommonRoadDatasetConfig
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import TemporalTrafficExtractorOptions
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TemporalTrafficExtractorFactory, \
    TrafficExtractorFactory
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from commonroad_geometric.simulation.simulation_factory import SimulationFactory


# Tutorials demonstrate how CommonRoad-Geometric should be used.
# Do not modify this for your own purposes. Create a tool or project instead.
def collect_temporal_dataset(
    raw_dir: Path,
    processed_dir: Path,
    samples_per_scenario: int,
) -> CommonRoadDataset:
    temporal_collector = TemporalDatasetCollector(
        extractor_factory=TemporalTrafficExtractorFactory(
            traffic_extractor_factory=TrafficExtractorFactory(
                options=TrafficExtractorOptions(
                    edge_drawer=VoronoiEdgeDrawer(dist_threshold=50)
                )),
            options=TemporalTrafficExtractorOptions(
                collect_num_time_steps=10,
                return_incomplete_temporal_graph=True
            )
        ),
        simulation_factory=SimulationFactory(
            options=ScenarioSimulationOptions()
        )
    )
    dataset: CommonRoadDataset[CommonRoadDataTemporal, CommonRoadDataTemporal] = CommonRoadDataset(
        config=CommonRoadDatasetConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_samples_per_scenario=samples_per_scenario,
            pre_processing_workers=2,
            pre_transform_workers=2
        ),
        collector=temporal_collector
    )
    # If the dataset has already been processed, this is a no-op
    dataset.process()

    # Without post load transform
    data = dataset.get(0)
    print(f"{data=}")
    # With post load transform
    transformed_data = dataset[0]
    print(f"{transformed_data=}")

    # Supports Pythonic way of accessing the last element
    last_data = dataset[-1]
    print(f"{last_data=}")
    # Supports slicing when using __getitem__
    dataset_slice: CommonRoadDataset = dataset[0:10]
    print(f"{dataset_slice=}")

    return dataset


if __name__ == '__main__':
    dataset_ = collect_temporal_dataset(
        raw_dir=Path("data/highd-sample"),
        processed_dir=Path("tutorials/output/dataset_t40"),
        samples_per_scenario=10,
    )
    print(f"Created dataset with {len(dataset_)} samples")
