from typing import Iterable, Tuple

import numpy as np
import torch
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from torch import Tensor


def average_road_coverage(dataset: CommonRoadDataset) -> Tensor:
    size = dataset[0].vehicle.road_coverage.size()[1:]
    accumulated_road_coverage = torch.zeros(size, dtype=torch.float)
    num_samples = 0
    for data in dataset:
        accumulated_road_coverage += data.vehicle.road_coverage.sum(dim=0)
        num_samples += data.vehicle.road_coverage.size(0)
    avg_road_coverage = accumulated_road_coverage / num_samples
    return avg_road_coverage


def median_road_coverage(dataset: CommonRoadDataset, avg_road_coverage: Tensor) -> Tensor:
    """Returns the sample which is closest to the average road coverage"""
    min_diff, min_tensor = float("inf"), None
    for data in dataset:
        diff = torch.tensor([
            road_coverage_diff(road_coverage, avg_road_coverage, aggr="abs")
            for road_coverage in data.vehicle.road_coverage
        ])
        min_diff_data, min_idx_data = diff.min(dim=0)
        if min_diff_data < min_diff:
            min_diff, min_tensor = min_diff_data, data.vehicle.road_coverage[min_idx_data]
    return min_tensor


def road_coverage_diff(road_coverage: Tensor, avg_road_coverage: Tensor, aggr: str = "abs") -> Tensor:
    diff = road_coverage - avg_road_coverage
    if aggr == "abs":
        return diff.abs_().sum(dim=(-2, -1))
    elif aggr == "square":
        return diff.square_().sum(dim=(-2, -1))
    else:
        raise TypeError(f"Unknown aggr {aggr}")


def total_lanelet_count(dataset: CommonRoadDataset) -> int:
    return sum(data.lanelet.num_nodes for data in dataset)


def total_lanelet_length(dataset: CommonRoadDataset) -> float:
    return sum(data.lanelet.x[:, 0].sum() for data in dataset)


def avg(xs: Iterable[float]) -> float:
    num = 0
    n = 0.0
    for x in xs:
        num += 1
        n += x
    return n / num if num > 0 else np.nan


def avg_total_lanelet_length_per_sample(dataset: CommonRoadDataset) -> float:
    return avg(data.lanelet.x[:, 0].sum() for data in dataset)


def lanelet_length_stats(dataset: CommonRoadDataset) -> Tuple[float, float, float, float]:
    lanelet_lengths = []
    for data in dataset:
        lanelet_lengths += data.lanelet.x[:, 0].tolist()
    mean = np.mean(lanelet_lengths).item()
    median = np.median(lanelet_lengths).item()
    min_len, max_len = min(lanelet_lengths), max(lanelet_lengths)
    return min_len, max_len, mean, median
