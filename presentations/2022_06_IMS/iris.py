import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biosphere import RandomForest
from changeforest import changeforest

from changeforest_simulations import load
from changeforest_simulations.constants import FIGURE_FONT_SIZE

plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

iris = load("iris")
variables = ["sepallength", "sepalwidth"]

X1 = iris.loc[lambda x: x["class"] == "Iris-versicolor", variables].to_numpy()
X2 = pd.concat(
    [
        iris.loc[lambda x: x["class"] == "Iris-versicolor", variables].sample(
            20, random_state=0
        ),
        iris.loc[lambda x: x["class"] == "Iris-setosa", variables].sample(
            30, random_state=0
        ),
    ]
).to_numpy()

n = X1.shape[0]


# Plot 1: Homogeneous time series
fig, axes = plt.subplots(nrows=X1.shape[1], figsize=(6, 3))
for idx in range(len(axes)):
    axes[idx].plot(X1[:, idx], color="black")
    if idx < len(axes) - 1:
        axes[idx].set_xticklabels([])
    ymin, ymax = axes[idx].get_ylim()
    axes[idx].vlines(25, ymin, ymax, color="blue", linestyles="dashed")
    axes[idx].set_ylabel(variables[idx])

plt.tight_layout()
plt.savefig("applications/iris_homogeneous.eps", dpi=300)


# Plot 2: Predictions from homogeneous time series
result = changeforest(X1, "random_forest", "bs")
fig, axes = plt.subplots(nrows=1, figsize=(6, 1.8))
axes.scatter(
    np.arange(len(X2)),
    result.optimizer_result.gain_results[1].predictions,
    color="black",
)
axes.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("applications/iris_predictions_homogeneous.eps", dpi=300)


# Plot 3: Heterogeneous time series
fig, axes = plt.subplots(nrows=X2.shape[1], figsize=(6, 3))
for idx in range(len(axes)):
    axes[idx].plot(X2[:, idx], color="black")
    if idx < len(axes) - 1:
        axes[idx].set_xticklabels([])
    ymin, ymax = axes[idx].get_ylim()
    axes[idx].vlines(20, ymin, ymax, color="green", linestyles="dashed")
    # axes[idx].vlines(25, ymin, ymax, color="blue", linestyles="dashed")
    axes[idx].set_ylabel(variables[idx])

plt.tight_layout()
plt.savefig("applications/iris_heterogeneous.eps", dpi=300)


# Plot 4: Predictions for heterogeneous time series
result = changeforest(X2, "random_forest", "bs")

fig, axes = plt.subplots(nrows=1, figsize=(6, 1.8))
axes.scatter(
    np.arange(len(X2)),
    result.optimizer_result.gain_results[1].predictions,
    color="black",
)
axes.vlines(20, 0.05, 0.95, color="green", linestyles="dashed")
axes.vlines(25, 0.05, 0.95, color="blue", linestyles="dashed")

axes.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("applications/iris_predictions_heterogeneous.eps", dpi=300)


def log_eta(x):
    eta = 0.0024787521766663585
    return np.log(eta + (1 - eta) * x)


# Plot 5: Full heterogeneous gain curve
def rf_gain(s, X):
    n = X.shape[0]
    y = np.zeros(n, dtype="float")
    y[0:s] = 1
    oob_predictions = RandomForest(max_depth=8).fit_predict_oob(X, y)
    log_likelihoods = np.zeros(n, dtype="float")

    log_likelihoods[0:s] = log_eta(n / (s - 1) * oob_predictions[0:s])
    log_likelihoods[s:n] = log_eta(n / (n - s - 1) * (1 - oob_predictions[s:n]))

    return log_likelihoods.sum()


gain = [np.nan, np.nan] + [rf_gain(s, X2) for s in range(2, n - 2)] + [np.nan, np.nan]

fig, axes = plt.subplots(nrows=1, figsize=(6, 1.8))
axes.plot(gain, color="black")
ymin, ymax = axes.get_ylim()
axes.vlines(20, ymin, ymax, color="green", linestyles="dashed")
axes.set_ylabel("gain")
axes.set_xlabel("s")

plt.tight_layout()
plt.savefig("applications/iris_gain_heterogeneous.eps", dpi=300)


# Plot 6: Approx heterogeneous gain curve
fig, axes = plt.subplots(nrows=1, figsize=(6, 1.8))
axes.plot(result.optimizer_result.gain_results[1].gain, color="black")
ymin, ymax = axes.get_ylim()
axes.vlines(20, ymin, ymax, color="green", linestyles="dashed")
axes.vlines(25, ymin, ymax, color="blue", linestyles="dashed")

plt.tight_layout()
plt.savefig("applications/iris_approx_gain_heterogeneous.eps", dpi=300)


# Plot 7: Two-Step-Search
plt.rcParams.update({"font.size": 8})
result.optimizer_result.plot()
plt.tight_layout()
plt.savefig("applications/iris_two_step_search.eps", dpi=300)

# Plot 8, 9: Pseudo Permutation test
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 14})
predictions = result.optimizer_result.gain_results[1].predictions

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
axes[0].scatter(np.arange(n), predictions, color="black")
axes[0].set_ylabel("original\npredictions")
axes[1].plot(result.optimizer_result.gain_results[1].gain, color="black")
axes[1].set_ylabel("original gain")
plt.tight_layout()
plt.savefig("applications/pseudo_permutation_test_1.eps", dpi=300)
print(f"Original gain: {np.max(result.optimizer_result.gain_results[1].gain):.2f}.")

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 6))

for idx in range(3):
    rng.shuffle(predictions)
    log_likelihoods = np.stack(
        [log_eta(2 * predictions), log_eta(2 * (1 - predictions))]
    )
    gain = log_likelihoods[1, :].sum() + np.concatenate(
        [[0], np.cumsum(log_likelihoods[0, :-1] - log_likelihoods[1, :-1])]
    )
    axes[idx, 0].scatter(np.arange(n), predictions, color="black")
    axes[idx, 0].set_ylabel("permuted\npredictions")
    axes[idx, 1].plot(gain, color="black")
    axes[idx, 1].set_ylabel("perm. gain")
    print(f"Iteration {idx} max gain: {np.max(gain):.2f}.")
    if idx < axes.shape[0] - 1:
        axes[idx, 0].set_xticklabels([])
        axes[idx, 1].set_xticklabels([])

plt.tight_layout(w_pad=4)
plt.savefig("applications/pseudo_permutation_test_2.eps", dpi=300)
