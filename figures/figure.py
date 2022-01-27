import click
import numpy as np
import pandas as pd
import plotnine as pn
from changeforest import Control, changeforest
from matplotlib import gridspec, pyplot

from changeforest_simulations import adjusted_rand_score, simulate


@click.command()
@click.option("--method", default="random_forest")
@click.option("--dataset", default="iris")
@click.option("--seed", default=0)
@click.option("--max-depth", default=5)
def main(method, dataset, seed, max_depth):
    changepoints, time_series = simulate(dataset, seed)
    if "repeated" in dataset:
        control = Control(minimal_relative_segment_length=0.001)
    else:
        control = Control(minimal_relative_segment_length=0.01)

    result = changeforest(time_series, method, "bs", control)

    nodes = [result]
    plots = []
    n = result.stop

    for _ in range(0, max_depth):
        new_nodes = []
        gains = []

        if len(nodes) == 0:
            break

        plots.append(
            pn.ggplot()
            + pn.geom_vline(
                xintercept=changepoints[1:-1], colour="green", linetype="dashed"
            )
        )
        # plots.append(pn.ggplot() + pn.geom_vline(xintercept=true_changepoints, colour="green", linetype="dashed"))
        for node in nodes:
            if node.optimizer_result is not None:
                gain = np.full(n, np.nan)
                gain[node.start : node.stop] = node.optimizer_result.gain_results[
                    -1
                ].gain
                gains.append(gain)

                plots[-1] += pn.geom_line(
                    data=pd.DataFrame({"t": range(n), "gain": gain}),
                    mapping=pn.aes(x="t", y="gain"),
                )

            if node.model_selection_result.is_significant:

                plots[-1] += pn.geom_vline(xintercept=node.best_split, colour="red")

            if node.left is not None:
                new_nodes.append(node.left)
            if node.right is not None:
                new_nodes.append(node.right)

        nodes = new_nodes

    # https://github.com/has2k1/plotnine/issues/373
    pn.options.figure_size = (10, 2 + 2 * len(plots))
    fig = (pn.ggplot() + pn.geom_blank(data=pd.DataFrame()) + pn.theme_void()).draw()
    gs = gridspec.GridSpec(len(plots), 1)

    estimated_changepoints = [0] + list(result.split_points()) + [len(time_series)]
    score = adjusted_rand_score(changepoints, estimated_changepoints)

    for i, plot in enumerate(plots):
        if i < len(plots) - 1:
            plot += pn.theme(axis_text_x=pn.element_blank())

        ax = fig.add_subplot(gs[i, 0])
        plot._draw_using_figure(fig, [ax])
        fig.suptitle(f"{method}: {dataset} (ARI={score:.2f})")

    pyplot.tight_layout()
    pyplot.savefig(f"figures/{dataset}_{method}_{seed}.png", dpi=300)

    print(f"True changeponts: {changepoints}.")
    print(f"Estimated changepoints: {estimated_changepoints}.")
    print(f"score: {score}.")


if __name__ == "__main__":
    main()
