import numpy as np
from changeforest import Control, changeforest
from matplotlib import pyplot as plt

from changeforest_simulations import simulate

alpha, X = simulate("iris", seed=0)

n = X.shape[0]

result = changeforest(
    X, "random_forest", "bs", Control(minimal_relative_segment_length=0.001)
)
gain_results = result.optimizer_result.gain_results

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 6))

for idx in range(3):
    axes[idx, 0].plot(range(n), gain_results[idx].gain, "k")
    ymin, ymax = axes[idx, 0].get_ylim()
    axes[idx, 0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, color="green")
    axes[idx, 0].vlines(
        np.nanargmax(gain_results[idx].gain),
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        color="red",
    )
    axes[idx, 0].vlines(
        gain_results[idx].guess, ymin=ymin, ymax=ymax, linestyles="dotted", color="blue"
    )
    axes[idx, 0].set_ylabel("approx. gain")

    axes[idx, 1].scatter(range(n), gain_results[idx].predictions, s=2, c="k")
    axes[idx, 1].set_ylabel("proba. prediction")

axes[2, 0].set_xlabel("s")
axes[2, 1].set_xlabel("s")

plt.tight_layout(pad=1.5)
plt.savefig("figures/two_step_search.eps", dpi=300)
