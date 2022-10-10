from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from misc import get_data, rss

from changeforest_simulations.constants import FIGURE_FONT_SIZE

plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

figures = Path(__file__).parents[1] / "figures"

## intro_one_changepoint.png
fig, ax = plt.subplots(figsize=(6, 4))

changepoints = np.array([0, 80, 200])
means = [-1, 2]
X = get_data(changepoints, means)

ax.plot(X, color="black")

for (start, stop), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
    ax.hlines(mean, start, stop, color="red", linestyles="dashed")

ymax, ymin = plt.ylim()
ax.vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")
fig.tight_layout()
fig.savefig(figures / "intro_one_changepoint.png", dpi=300)

## intro_one_changepoint_rss_None.png
fig, ax = plt.subplots(figsize=(6, 4))

changepoints = np.array([0, 80, 200])
N = changepoints[-1]
means = [-1, 2]
X = get_data(changepoints, means)

ax.plot(X, color="black")

for (start, stop), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
    plt.hlines(mean, start, stop, color="red", linestyles="dashed")

ymax, ymin = plt.ylim()
ax.vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")

ax.hlines(np.mean(X), 0, N, color="blue")
ax.fill_between(np.arange(0, N), np.mean(X[0:N]), X[0:N], color="blue", alpha=0.2)
ax.text(0, 3, f"RSS={rss(X, 0):.2f}", color="black")
fig.tight_layout()
fig.savefig(figures / "intro_one_changepoint_rss_none.png", dpi=300)

print(f"X.mean() = {X.mean():.2f}")


## intro_one_changepoint_colored_None.png
s = 50
fig, ax = plt.subplots(figsize=(6, 4))

changepoints = np.array([0, 80, 200])
N = changepoints[-1]
means = [-1, 2]
X = get_data(changepoints, means)

ax.plot(X, color="black")

ymax, ymin = plt.ylim()
ax.vlines([s], ymin, ymax, linestyles="dashed", color="blue")
ax.hlines(np.mean(X), 0, N, color="blue")
ax.fill_between(np.arange(0, s), np.mean(X[0:N]), X[0:s], color="red", alpha=0.2)
ax.fill_between(np.arange(s, N), np.mean(X[0:N]), X[s:N], color="cyan", alpha=0.2)
fig.tight_layout()
fig.savefig(figures / "intro_one_changepoint_colored_none.png", dpi=300)

## intro_one_changepoint_rss_{s}.png
for s in [50, 100, 150]:
    fig, ax = plt.subplots(figsize=(6, 4))

    changepoints = np.array([0, 80, 200])
    N = changepoints[-1]
    means = [-1, 2]
    X = get_data(changepoints, means)

    ax.plot(X, color="black")

    for (start, stop), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
        plt.hlines(mean, start, stop, color="red", linestyles="dashed")

    ymin, ymax = plt.ylim()
    ax.vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")

    ax.vlines([s], ymin, ymax, linestyles="dashed", color="blue")
    ax.hlines(np.mean(X[0:s]), 0, s, color="blue")
    ax.hlines(np.mean(X[s:N]), s, N, color="blue")
    ax.fill_between(np.arange(0, s), np.mean(X[0:s]), X[0:s], color="blue", alpha=0.2)
    ax.fill_between(np.arange(s, N), np.mean(X[s:N]), X[s:N], color="blue", alpha=0.2)
    ax.text(0, 3, f"RSS={rss(X, s):.2f}", color="black")
    ax.text(s + 3, ymin, f"s={s}", color="blue")
    fig.tight_layout()
    fig.savefig(figures / f"intro_one_changepoint_rss_{s}.png", dpi=300)

    print(
        f"s={s}, X[:s].mean() = {X[:s].mean():.2f}, X[s:].mean() = {X[s:].mean():.2f}"
    )


# intro_one_changepoint_colored_50.png
s = 50
fig, ax = plt.subplots(figsize=(6, 4))

changepoints = np.array([0, 80, 200])
N = changepoints[-1]
means = [-1, 2]
X = get_data(changepoints, means)

ax.plot(X, color="black")

ymin, ymax = plt.ylim()

ax.vlines([s], ymin, ymax, linestyles="dashed", color="blue")
ax.hlines(np.mean(X[0:s]), 0, s, color="blue")
ax.hlines(np.mean(X[s:N]), s, N, color="blue")
ax.fill_between(np.arange(0, s), np.mean(X[0:s]), X[0:s], color="green", alpha=0.2)
ax.fill_between(np.arange(s, N), np.mean(X[s:N]), X[s:N], color="orange", alpha=0.2)
fig.tight_layout()
fig.savefig(figures / f"intro_one_changepoint_colored_{s}.png", dpi=300)

# intro_one_changepoint_gain.png
gain = rss(X, 0) - np.array([rss(X, s) for s in range(0, N)])
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(gain, color="black")
fig.savefig(figures / "intro_one_changepoint_gain.png", dpi=300)
