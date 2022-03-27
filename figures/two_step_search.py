import numpy as np
from biosphere import RandomForest
from changeforest import Control, changeforest
from matplotlib import pyplot as plt

from changeforest_simulations import simulate
from changeforest_simulations.constants import (
    COLORS,
    DOTTED_LINEWIDTH,
    FIGURE_FONT_SIZE,
    FIGURE_WIDTH,
    X_MARKER_KWARGS,
)

red = COLORS["red"]
green = COLORS["green"]
blue = COLORS["blue"]
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

alpha, X = simulate("change_in_covariance", seed=2)

n = X.shape[0]

result = changeforest(
    X, "random_forest", "bs", Control(minimal_relative_segment_length=0.001)
)

gain_results = result.optimizer_result.gain_results
gain_results[0:3] = sorted(gain_results[0:3], key=lambda x: x.guess)

min_gain = min(min(gain_result.gain) for gain_result in gain_results)
max_gain = max(max(gain_result.gain) for gain_result in gain_results)
delta = max_gain - min_gain
(ymin_gain, ymax_gain) = (min_gain - 0.05 * delta, max_gain + 0.05 * delta)
gain_range = (min_gain - 0.1 * delta, max_gain + 0.1 * delta)

ymin_proba = -0.05
ymax_proba = 1.05
proba_range = (-0.1, 1.1)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 2 / 3))

for idx in range(4):
    axes[idx, 0].plot(range(n), gain_results[idx].gain, color="k")
    axes[idx, 0].set_ylim(*gain_range)

    axes[idx, 0].vlines(
        gain_results[idx].guess,
        ymin=ymin_gain,
        ymax=ymax_gain,
        linestyles="dashed",
        color=blue,
        linewidth=DOTTED_LINEWIDTH,
    )
    axes[idx, 0].vlines(
        np.nanargmax(gain_results[idx].gain),
        ymin=ymin_gain,
        ymax=ymax_gain,
        linestyles="dotted",
        color=red,
        linewidth=DOTTED_LINEWIDTH,
    )
    axes[idx, 0].scatter(alpha[1:-1], [ymin_gain] * (len(alpha) - 2), **X_MARKER_KWARGS)
    axes[idx, 0].set_ylabel("approx. gain")

    axes[idx, 1].set_ylim(*proba_range)
    axes[idx, 1].vlines(
        gain_results[idx].guess,
        ymin=ymin_proba,
        ymax=ymax_proba,
        linestyles="dashed",
        color=blue,
        linewidth=DOTTED_LINEWIDTH,
    )

    axes[idx, 1].scatter(
        alpha[1:-1], [ymin_proba] * (len(alpha) - 2), **X_MARKER_KWARGS
    )
    axes[idx, 1].scatter(range(n), gain_results[idx].predictions, s=2, c="k")
    axes[idx, 1].set_ylabel("proba. predictions")
    if idx < 3:
        axes[idx, 0].set_xticklabels([])
        axes[idx, 1].set_xticklabels([])

axes[-1, 0].set_xlabel("split")
axes[-1, 1].set_xlabel("t")

fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])

# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig("figures/two_step_search.png", dpi=300)
plt.savefig("figures/two_step_search.eps", dpi=300)

# two_step_search_biased.png
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 5))

y = np.zeros(n, dtype="float")
s = int(3 * X.shape[0] / 4)

y[0:s] = 1
rf = RandomForest()
rf.fit(X, y)
predictions = rf.predict(X)  # not _oob

log_likelihoods = np.stack(
    [
        np.log(0.0024787521766663585 + 0.9975212478233336 * n / s * predictions),
        np.log(
            0.0024787521766663585 + 0.9975212478233336 * n / (n - s) * (1 - predictions)
        ),
    ]
)

gain = log_likelihoods[1, :].sum() + np.concatenate(
    [[0], np.cumsum(log_likelihoods[0, :-1] - log_likelihoods[1, :-1])]
)

axes[0].plot(range(n), gain, "k")
ymin, ymax = axes[0].get_ylim()
# axes[0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, color=green)
axes[0].scatter(alpha[1:-1], [ymin] * (len(alpha) - 2), **X_MARKER_KWARGS)
axes[0].vlines(
    s, ymin=ymin, ymax=ymax, linestyles="dashed", color=blue, linewidth=DOTTED_LINEWIDTH
)
axes[0].vlines(
    np.nanargmax(gain),
    ymin=ymin,
    ymax=ymax,
    linestyles="dotted",
    color=red,
    linewidth=DOTTED_LINEWIDTH,
)
axes[0].set_ylabel("approx. gain")
axes[0].set_xlabel("split")

axes[1].scatter(range(n), predictions, s=2, c="k")
axes[1].set_ylabel("proba. predictions")
axes[1].set_xlabel("t")
axes[1].set_ylim(-0.1, 1.1)
ymin = -0.05
ymax = 1.05
axes[1].vlines(
    s, ymin=ymin, ymax=ymax, linestyles="dashed", color=blue, linewidth=DOTTED_LINEWIDTH
)
axes[1].scatter(alpha[1:-1], [ymin] * (len(alpha) - 2), **X_MARKER_KWARGS)

plt.tight_layout()
plt.savefig("figures/two_step_search_biased.eps", dpi=300)
plt.savefig("figures/two_step_search_biased.png", dpi=300)
