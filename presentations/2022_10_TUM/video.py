from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from misc import get_data, rss

from changeforest_simulations.constants import FIGURE_FONT_SIZE

plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

figures = Path(__file__).parents[1] / "figures"

changepoints = [0, 80, 200]
means = [-1, 2]

X = get_data(changepoints, means)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].plot(X, color="black")

for (start, stop), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
    axes[0].hlines(mean, start, stop, color="red", linestyles="dashed")

ymin, ymax = axes[0].get_ylim()
axes[0].vlines(changepoints[1:-1], ymin, ymax, linestyles="dashed", color="green")
vertical_s = axes[0].vlines([], ymin, ymax, linestyles="dashed", color="blue")
horizontal_left = axes[0].hlines([], [], [], color="blue")
horizontal_right = axes[0].hlines([], [], [], color="blue")
fill = axes[0].fill_between(np.arange(0, 200), 0, X, color="blue", alpha=0)
rss_text = axes[0].text(0, 3, "", color="black")
s_text = axes[0].text(170, ymin, "", color="blue")

split_candidates = list(range(10, 190))
gains = rss(X, 0) - np.array([rss(X, guess) for guess in split_candidates])
axes[1].set_xlim(0, 200)
axes[1].plot(split_candidates, gains, color="black")
gain_dot = axes[1].scatter([], [], color="blue", s=40)


def get_vertices(X, frame):
    # From https://stackoverflow.com/questions/16120801/matplotlib-animate-fill-between-shape
    # and reverse engineering. I don't know what I'm doing. The 3 extra rows are there
    # to align with fill.get_paths()[0].codes
    N = len(X)
    a = np.zeros((2 * N + 3, 2), dtype=np.float64)
    a[1 : (N + 1), 0] = np.arange(N)
    a[(N + 2) : (2 * N + 2), 0] = np.arange(N)[::-1]
    a[(N + 2) : -1, 1] = X[::-1]
    a[: (frame + 1), 1] = np.mean(X[:frame])
    a[(frame + 1) : (N + 3), 1] = np.mean(X[frame:])
    a[0, :] = a[1, :]
    a[N + 1, :] = a[N + 2, :]
    a[-1, :] = a[-2, :]
    return a


def animate(frame):
    gain = rss(X, 0) - rss(X, frame)
    vertical_s.set_segments([np.array([[frame, ymin], [frame, ymax]])])
    horizontal_left.set_segments(
        [np.array([[0, np.mean(X[0:frame])], [frame, np.mean(X[0:frame])]])]
    )
    horizontal_right.set_segments(
        [np.array([[frame, np.mean(X[frame:])], [200, np.mean(X[frame:])]])]
    )

    path = fill.get_paths()[0]
    path.vertices = get_vertices(X, frame)
    fill.set_alpha(0.2)
    rss_text.set_text(f"gain: {gain:.2f}")
    s_text.set_text(f"s: {frame}")
    gain_dot.set_offsets(np.array([[frame, gain]]))
    return (
        vertical_s,
        horizontal_left,
        horizontal_right,
        fill,
        rss_text,
        s_text,
        gain_dot,
    )


ani = animation.FuncAnimation(fig, animate, split_candidates, interval=40, blit=True)

ani.save(figures / "video.mp4", fps=30, dpi=300)
