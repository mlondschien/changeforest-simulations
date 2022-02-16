import numpy as np
from biosphere import RandomForest
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from changeforest_simulations import simulate

alpha, X = simulate("iris")
n = X.shape[0]


def rf_gain(s, X):
    n = X.shape[0]
    y = np.zeros(n, dtype="float")
    y[0:s] = 1
    oob_predictions = RandomForest().fit_predict_oob(X, y)

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
    predictions = KNeighborsClassifier().fit(X, y).predict(X)

    log_likelihoods = np.zeros(n, dtype="float")
    log_likelihoods[0:s] = np.log(
        0.0024787521766663585 + 0.9975212478233336 * n / (s - 1) * predictions[0:s]
    )
    log_likelihoods[s:n] = np.log(
        0.0024787521766663585
        + 0.9975212478233336 * n / (n - s - 1) * (1 - predictions[s:n])
    )

    return log_likelihoods.sum()


knn_gain_curve = (
    [np.nan, np.nan] + [knn_gain(s, X) for s in range(2, n - 2)] + [np.nan, np.nan]
)

fig, axes = plt.subplots(ncols=2, figsize=(9, 3.5))

axes[0].plot(range(n), rf_gain_curve, "k")
ymin, ymax = axes[0].get_ylim()
axes[0].vlines(alpha[1:-1], ymin=ymin, ymax=ymax, linestyles="solid", color="green")
axes[0].vlines(
    np.nanargmax(rf_gain_curve),
    ymin=np.nanmin(rf_gain_curve),
    ymax=np.nanmax(rf_gain_curve),
    linestyles="dashed",
    color="red",
)
axes[0].set_xlabel("s")
axes[0].set_ylabel("gain")

axes[1].plot(range(n), knn_gain_curve, "k")
ymin, ymax = axes[1].get_ylim()
axes[1].vlines(alpha[1:-1], ymin=ymax, ymax=ymin, linestyles="solid", color="green")
axes[1].vlines(
    np.nanargmax(knn_gain_curve), ymin=ymin, ymax=ymax, linestyles="dashed", color="red"
)
axes[1].set_xlabel("s")
# axes[1].set_ylabel("gain")

plt.tight_layout()
plt.savefig("figures/gain.png", dpi=300)
