from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader

from commonroad_geometric.common.torch_utils.sampling import sample_indices
from commonroad_geometric.dataset.commonroad_dataset import CommonRoadDataset
from projects.geometric_models.drivable_area.utils.visualization.plotting import plot_road_coverage, plot_with_axes


def visualize_road_coverage(
    dataset: CommonRoadDataset,
    *,
    grid: Tuple[int, int] = (4, 4),
    figsize: Tuple[int, int] = (16, 16),
    batch_size: int = 10,
    shuffle: bool = True
) -> None:
    num_samples = grid[0] * grid[1]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    for data in loader:
        indices_list = [
            ('weighted', sample_indices(data.sampling_weights, num_samples=num_samples)),
            ('random', torch.randint(data.v.num_nodes, size=(num_samples,)))
        ]

        print(f"Visualizing batch with {data.num_graphs=}, {data.scenario_id=}, {data.time_step=}")

        for title, indices in indices_list:
            fig, axs = plt.subplots(*grid, figsize=figsize)
            axs_list = list((i, ax) for i, ax in enumerate(axs.flat))
            for i, ax in axs_list:
                print(i, '/', len(axs_list))
                with plot_with_axes(axes=ax):
                    idx = indices[i]
                    weight = data.sampling_weights[idx]
                    plot_road_coverage(data.v.drivable_area[idx], title=f"{data.v.batch[idx]} ({title}-{weight:.2f})")
            fig.tight_layout()

        plt.show()
