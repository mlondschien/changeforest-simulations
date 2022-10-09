from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from misc import get_data, rss

from changeforest_simulations.constants import FIGURE_FONT_SIZE

plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

figures = Path(__file__).parents[1] / "figures"

N = 200
changepoints = np.array([0, 25, 70, 140, 200])
means = [-2, 0, -4, -2]

X = get_data(changepoints, means, 1)


def plot(start, stop, changepoints, means, ticks=None):
    width = 3 + (stop - start) / 30
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width, 3))
    axes[0].plot(np.arange(start, stop), X[start:stop], color="black")

    for (start_, stop_), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
        axes[0].hlines(mean, start_, stop_, color="red", linestyles="dashed")

    ymax, ymin = axes[0].get_ylim()
    axes[0].vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")

    gain = rss(X[start:stop], 0) - np.array(
        [rss(X[start:stop], guess) for guess in range(0, stop - start)]
    )
    best_split = np.argmax(gain) + start
    axes[1].plot(np.arange(start, stop), gain, color="black")
    ymin, ymax = axes[1].get_ylim()
    axes[1].vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")
    axes[1].vlines([best_split], ymin, ymax, color="red", linestyles="dotted")

    if ticks is not None:
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(ticks)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels(ticks)

    fig.tight_layout()

    return fig, best_split


fig, best_split = plot(0, 200, changepoints, means)
fig.savefig(figures / "binary_segmentation_0.png", dpi=300)

fig, _ = plot(
    0,
    best_split,
    [c for c in changepoints if c < best_split] + [best_split],
    [z[0] for z in zip(means, changepoints) if z[1] < best_split],
    ticks=[0, 50, 70],
)
fig.savefig(figures / "binary_segmentation_1.png", dpi=300)

fig, _ = plot(
    best_split,
    200,
    [best_split] + [c for c in changepoints if c > best_split],
    [z[0] for z in zip(means, changepoints) if z[1] >= best_split],
    ticks=[70, 100, 150, 200],
)
fig.savefig(figures / "binary_segmentation_2.png", dpi=300)
