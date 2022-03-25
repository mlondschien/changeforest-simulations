import numpy as np
from biosphere import RandomForest
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from changeforest_simulations import simulate
from changeforest_simulations.constants import (
    COLORS,
    DOTTED_LINEWIDTH,
    FIGURE_FONT_SIZE,
    FIGURE_WIDTH,
)

alpha, X = simulate("change_in_covariance", seed=2)
n = X.shape[0]

red = COLORS["red"]
green = COLORS["green"]
blue = COLORS["blue"]
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})


def rf_gain(s, X):
    n = X.shape[0]
    y = np.zeros(n, dtype="float")
    y[0:s] = 1
    oob_predictions = RandomForest(
        max_depth=8, n_estimators=100, n_jobs=-1, max_features="sqrt"
    ).fit_predict_oob(X, y)

    log_likelihoods = np.zeros(n, dtype="float")
    log_likelihoods[0:s] = np.log(
        0.0024787521766663585 + 0.9975212478233336 * n / (s - 1) * oob_predictions[0:s]
    )
    log_likelihoods[s:n] = np.log(
        0.0024787521766663585
        + 0.9975212478233336 * n / (n - s - 1) * (1 - oob_predictions[s:n])
    )

    return log_likelihoods.sum()


rf_gain_curve = (
    [np.nan, np.nan] + [rf_gain(s, X) for s in range(2, n - 2)] + [np.nan, np.nan]
)


def knn_gain(s, X):
    n = X.shape[0]
    y = np.zeros(n, dtype="float")
    y[0:s] = 1
    predictions = (
        KNeighborsClassifier(n_jobs=1, n_neighbors=int(np.sqrt(n)))
        .fit(X, y)
        .predict_proba(X)[:, 1]
    )
    log_likelihoods = np.zeros(n, dtype="float")
    log_likelihoods[0:s] = np.log(
        0.0024787521766663585 + 0.9975212478233336 * n / s * predictions[0:s]
    )
    log_likelihoods[s:n] = np.log(
        0.0024787521766663585
        + 0.9975212478233336 * n / (n - s) * (1 - predictions[s:n])
    )

    return log_likelihoods.sum()


knn_gain_curve = (
    [np.nan, np.nan] + [knn_gain(s, X) for s in range(2, n - 2)] + [np.nan, np.nan]
)

fig, axes = plt.subplots(ncols=2, figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 5))

axes[0].plot(range(n), rf_gain_curve, "k")
ymin, ymax = axes[0].get_ylim()
# axes[0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, linestyles="solid", color=green)
axes[0].vlines(
    np.nanargmax(rf_gain_curve),
    ymin=np.nanmin(rf_gain_curve),
    ymax=np.nanmax(rf_gain_curve),
    linestyles="dotted",
    color=red,
    linewidth=DOTTED_LINEWIDTH,
)
ymin, _ = axes[0].get_ylim()
axes[0].scatter(
    alpha[1:-1], [ymin] * (len(alpha) - 2), marker="x", s=2, color=green, linewidth=10
)
axes[0].set_xlabel("split")
axes[0].set_ylabel("gain")

axes[1].plot(range(n), knn_gain_curve, "k")
ymin, ymax = axes[1].get_ylim()
axes[1].scatter(
    alpha[1:-1], [ymin] * (len(alpha) - 2), marker="x", s=2, color=green, linewidth=10
)
axes[1].vlines(
    np.nanargmax(knn_gain_curve),
    ymin=ymin,
    ymax=ymax,
    linestyles="dotted",
    color=red,
    linewidth=DOTTED_LINEWIDTH,
)
axes[1].set_xlabel("split")
# axes[1].set_ylabel("gain")

plt.tight_layout()
plt.savefig("figures/gains.eps", dpi=300)
plt.savefig("figures/gains.png", dpi=300)
