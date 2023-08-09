import functools
from typing import Union
import torch
from torch import Tensor
from typing import Callable
from commonroad_geometric.dataset.transformation.dataset_transformation import dataset_transformation
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from projects.geometric_models.drivable_area.utils.dataset import average_road_coverage, road_coverage_diff



def compute_weights_for_target_distribution(
    input_dist: Tensor,
    target_dist: Union[None, Tensor, Callable[[float], float]] = None,
    num_bins: int = 200,
):
    hist_input, bin_edges_input = torch.histogram(input_dist, bins=num_bins, density=False)
    hist_input /= hist_input.sum()
    hist_input[hist_input != 0] = 1 / hist_input[hist_input != 0]

    weight_distr = torch.log(hist_input + 1.0)

    target_dist_is_tensor = isinstance(target_dist, Tensor)
    if target_dist_is_tensor:
        hist_target, _ = torch.histogram(target_dist, bins=num_bins, density=True)

    weights = torch.empty_like(input_dist)
    for i in range(input_dist.size(0)):
        # find index of bin which contains input_dist[i]
        smaller_bin_edges = (bin_edges_input < input_dist[i]).nonzero()
        if smaller_bin_edges.size(0) == 0:
            bin_index = 0
        else:
            bin_index = smaller_bin_edges.max()

        if target_dist is None:
            # if target_dist is None the target distribution is a uniform distribution
            c = 1
        elif target_dist_is_tensor:
            c = hist_target[bin_index]
        else:
            c = target_dist(input_dist[i])

        weights[i] = c * weight_distr[bin_index]

    return weights



def add_sample_weights(
    dataset: CommonRoadDataset,
    lanelet_sampling_weights_num_bins: Union[int, float],
    num_workers: int = 1
) -> None:
    if isinstance(lanelet_sampling_weights_num_bins, float):
        lanelet_sampling_weights_num_bins = max(1, int(lanelet_sampling_weights_num_bins*len(dataset)))

    def _assign_lanelet_sampling_weight(
        scenario_index: int,
        sample_index: int,
        data,
        lanelet_sampling_weights,
    ):
        data.sampling_weights = lanelet_sampling_weights[sample_index]
        assert data.sampling_weights.ndim == 1
        assert data.sampling_weights.size(0) == data.v.num_nodes
        yield data
    # Lanelet pre-processing iteration 2: assign sampling weight to each lanelet

    avg_road_coverage = average_road_coverage(dataset)
    # med_road_coverage = median_road_coverage(dataset, avg_road_coverage)

    all_road_coverage_diffs = [
        road_coverage_diff(data.vehicle.road_coverage, avg_road_coverage, aggr="abs")
        for data in dataset
    ]

    # TODO alternatively with a (Gaussian) KDE
    #      https://scipy.github.io/devdocs/reference/generated/scipy.stats.gaussian_kde.html
    all_weights = compute_weights_for_target_distribution(
        input_dist=torch.cat(all_road_coverage_diffs, dim=0),
        target_dist=None,#lambda v: 1 if v < 200 else 0.1,  # TODO! distribution is arbitrary
        num_bins=lanelet_sampling_weights_num_bins,
    )
    assert sum(d.size(0) for d in all_road_coverage_diffs) == all_weights.size(0)

    weights = torch.split(all_weights, [d.size(0) for d in all_road_coverage_diffs])

    dataset_transformation(
        dataset=dataset,
        transform=functools.partial(
            _assign_lanelet_sampling_weight,
            lanelet_sampling_weights=weights
        ),
        num_workers=num_workers,
    )
