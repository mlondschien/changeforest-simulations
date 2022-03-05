import numpy as np
from biosphere import RandomForest
from changeforest import Control, changeforest
from matplotlib import pyplot as plt

from changeforest_simulations import simulate
from changeforest_simulations.constants import COLORS

red = COLORS["red"]
green = COLORS["green"]
blue = COLORS["blue"]
plt.rcParams.update({"font.size": 12})

alpha, X = simulate("iris", seed=1)

n = X.shape[0]

result = changeforest(
    X, "random_forest", "bs", Control(minimal_relative_segment_length=0.001)
)
gain_results = result.optimizer_result.gain_results

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 7))

for idx in range(4):
    axes[idx, 0].plot(range(n), gain_results[idx].gain, color="k")
    ymin, ymax = axes[idx, 0].get_ylim()
    axes[idx, 0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, color=green)
    axes[idx, 0].vlines(
        np.nanargmax(gain_results[idx].gain),
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        color=red,
    )
    axes[idx, 0].vlines(
        gain_results[idx].guess, ymin=ymin, ymax=ymax, linestyles="dotted", color=blue
    )
    axes[idx, 0].set_ylabel("approx. gain")

    axes[idx, 1].scatter(range(n), gain_results[idx].predictions, s=2, c="k")
    axes[idx, 1].set_ylabel("proba. prediction")

axes[-1, 0].set_xlabel("s")
axes[-1, 1].set_xlabel("t")

# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig("figures/two_step_search.png", dpi=300)
plt.savefig("figures/two_step_search.eps", dpi=300)


# two_step_search_biased.png
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))

y = np.zeros(n, dtype="float")
s = 75

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
axes[0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, color=green)
axes[0].vlines(
    np.nanargmax(gain), ymin=ymin, ymax=ymax, linestyles="dashed", color=red,
)
axes[0].vlines(s, ymin=ymin, ymax=ymax, linestyles="dotted", color=blue)
axes[0].set_ylabel("approx. gain")
axes[0].set_xlabel("s")

axes[1].scatter(range(n), predictions, s=2, c="k")
axes[1].set_ylabel("proba. prediction")
axes[1].set_xlabel("t")

plt.tight_layout()
plt.savefig("figures/two_step_search_biased.eps", dpi=300)
plt.savefig("figures/two_step_search_biased.png", dpi=300)
