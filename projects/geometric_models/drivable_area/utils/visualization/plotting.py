import contextlib
import typing as t

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor

_GLOBAL_AXES: t.Optional[Axes] = None


@contextlib.contextmanager
def plot_with_axes(axes: Axes):
    global _GLOBAL_AXES
    prev_ax = _GLOBAL_AXES
    try:
        _GLOBAL_AXES = axes
        yield
    finally:
        _GLOBAL_AXES = prev_ax



def plot_road_coverage(road_coverage: Tensor, *, title: t.Optional[str] = None) -> None:
    import seaborn as sns

    if (ax := _GLOBAL_AXES) is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    sns.heatmap(road_coverage.flipud(), linewidths=.5, cmap="Blues", ax=ax, vmin=0, vmax=1,
                cbar=False, xticklabels=False, yticklabels=False)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    if _GLOBAL_AXES is None:
        fig.tight_layout()
        plt.show()


def plot_distribution(
    vec: Tensor,
    min_val: t.Optional[float] = None,
    max_val: t.Optional[float] = None,
    num_bins: t.Optional[int] = None,
    title: t.Optional[str] = None,
) -> None:
    import seaborn as sns
    if min_val is None:
        min_val = vec.min().item()
    if max_val is None:
        max_val = vec.max().item()

    fig, axs = plt.subplots(nrows=2, figsize=(6, 12))
    if title:
        fig.suptitle(title, fontsize=16)


    sns.ecdfplot(data=vec, stat="count", ax=axs[0])
    sns.rugplot(x=vec, ax=axs[0])
    axs[0].grid(True, which="major")
    axs[0].set_title("Empirical cumulative distribution")

    # distribution histogram
    # https://seaborn.pydata.org/generated/seaborn.histplot.html
    # https://en.m.wikipedia.org/wiki/Kernel_density_estimation
    # log_scale = (False, True)
    log_scale = (False, False)
    sns.histplot(
        x=vec,
        bins=num_bins or 400,
        binrange=(min_val, max_val),
        log_scale=log_scale,
        kde=True,
        kde_kws=dict(bw_adjust=0.2),
        ax=axs[1],
    )
    axs[1].set_title("Histogram & Gaussian KDE")

    fig.tight_layout()
    plt.show()


def create_drivable_area_prediction_image(prediction: Tensor) -> Tensor:
    drivable_area_mask: Tensor = prediction > 0.5
    prediction_rgb = torch.empty((*prediction.size(), 3), dtype=torch.float, device="cpu")
    prediction_rgb[..., 1] = drivable_area_mask.type(torch.float) * prediction
    prediction_rgb[..., 0] = (~drivable_area_mask).type(torch.float) * prediction
    prediction_rgb[..., 2] = 0.0
    prediction_rgb = (prediction_rgb * 255).type(torch.uint8).cpu()
    return prediction_rgb


