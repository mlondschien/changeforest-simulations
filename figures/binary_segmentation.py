import click
import matplotlib.pyplot as plt
import numpy as np
from changeforest import Control, changeforest

from changeforest_simulations import adjusted_rand_score, simulate
from changeforest_simulations.constants import (
    COLORS,
    DOTTED_LINEWIDTH,
    FIGURE_FONT_SIZE,
    FIGURE_WIDTH,
    X_MARKER_KWARGS,
)


@click.command()
@click.option("--method", default="random_forest")
@click.option("--dataset", default="iris")
@click.option("--seed", default=0)
@click.option("--max-depth", default=5)
def main(method, dataset, seed, max_depth):
    changepoints, time_series = simulate(dataset, seed)
    n = time_series.shape[0]
    delta = 0.01
    minimal_segment_length = int(np.ceil(n * delta))

    control = Control(minimal_relative_segment_length=delta)
    result = changeforest(time_series, method, "bs", control)
    print(
        f"""\
True change points: {changepoints}
Found change points: {result.split_points()}
ARI: {adjusted_rand_score(changepoints, [0] + result.split_points() + [n])}"""
    )
    nodes = [result]
    gains = []
    splits = []
    found_changepoints = []
    guesses = []

    for _ in range(0, max_depth):
        new_nodes = []

        if len(nodes) == 0:
            break

        gains.append([])
        found_changepoints.append([])
        splits.append([])
        guesses.append([])

        for node in nodes:
            splits[-1].append(node.start)
            splits[-1].append(node.stop)

            if node.optimizer_result is not None:
                result = node.optimizer_result.gain_results[-1]
                gains[-1].append(np.full(n, np.nan))
                gains[-1][-1][
                    (node.start + minimal_segment_length) : (
                        node.stop - minimal_segment_length
                    )
                ] = result.gain[minimal_segment_length:-minimal_segment_length]
                if result.guess is not None:
                    guesses[-1].append(result.guess)

            if node.model_selection_result.is_significant:
                found_changepoints[-1].append(node.best_split)

            if node.left is not None:
                new_nodes.append(node.left)
            if node.right is not None:
                new_nodes.append(node.right)

        nodes = new_nodes

    depth = len(gains)
    plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

    fig, axes = plt.subplots(
        nrows=depth, figsize=(FIGURE_WIDTH, depth * FIGURE_WIDTH / 5)
    )
    if depth == 1:
        axes = [axes]

    for idx in range(depth):

        for gain in gains[idx]:
            axes[idx].plot(gain, color="black")

        ymin, ymax = axes[idx].get_ylim()
        new_ymin, new_ymax = ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)

        axes[idx].vlines(splits[idx], color="black", ymin=new_ymin, ymax=new_ymax)

        if guesses[idx]:
            axes[idx].vlines(
                guesses[idx],
                color=COLORS["blue"],
                ymin=ymin,
                ymax=ymax,
                linewidth=DOTTED_LINEWIDTH,
                linestyles="dashed",
            )

        axes[idx].vlines(
            found_changepoints[idx],
            color=COLORS["red"],
            ymin=ymin,
            ymax=ymax,
            linewidth=DOTTED_LINEWIDTH,
            linestyles="dotted",
        )

        axes[idx].scatter(
            changepoints[1:-1], [ymin] * (len(changepoints) - 2), **X_MARKER_KWARGS
        )

        axes[idx].set_xlim(0, n)
        axes[idx].set_ylim(new_ymin, new_ymax)
        axes[idx].set_ylabel("approx. gain")
        if idx < depth - 1:
            axes[idx].set_xticklabels([])

    axes[-1].set_xlabel("split")
    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.savefig(f"figures/binary_segmentation_{dataset}_{method}_{seed}.png", dpi=300)
    plt.savefig(f"figures/binary_segmentation_{dataset}_{method}_{seed}.eps", dpi=300)


if __name__ == "__main__":
    main()
