import json
from pathlib import Path

import click
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt

from changeforest_simulations import simulate
from changeforest_simulations.constants import (
    COLOR_CYCLE,
    FIGURE_FONT_SIZE,
    FIGURE_WIDTH,
    METHOD_ORDERING,
    METHOD_RENAMING,
    X_MARKER_KWARGS,
)

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "main"
figures_path = Path(__file__).parent

plt.rc("axes", prop_cycle=cycler(color=list(COLOR_CYCLE)))
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})


@click.command()
@click.option("--file", default=None)
@click.option("--dataset", default="dirichlet")
def main(file, dataset):
    alpha, _ = simulate(dataset)

    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")], axis=0)
    df["method"] = df["method"].replace(METHOD_RENAMING)

    df = df[lambda x: x["dataset"].eq(dataset)]
    if df.empty:
        raise ValueError(f"No entries for  dataset {dataset}.")

    changepoints = (
        df["estimated_changepoints"].apply(lambda x: json.loads(x)[1:-1]).sum()
    )

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(FIGURE_WIDTH, FIGURE_WIDTH))
    for idx, method in enumerate(METHOD_ORDERING[1:]):  # Exclude change in mean
        changepoints = (
            df.loc[lambda x: x["method"].eq(method), "estimated_changepoints"]
            .apply(lambda x: json.loads(x)[1:-1])
            .sum()
        )
        axes[idx // 2, idx % 2].hist(
            changepoints, bins=1000, color="black", range=(0, alpha[-1])
        )
        axes[idx // 2, idx % 2].set_title(method)
        _, ymax = axes[idx // 2, idx % 2].get_ylim()

        axes[idx // 2, idx % 2].scatter(
            alpha[1:-1], [ymax] * (len(alpha) - 2), **X_MARKER_KWARGS
        )

    fig.savefig(figures_path / f"histograms_{dataset}.eps", dpi=300)
    fig.savefig(figures_path / f"histograms_{dataset}.png", dpi=300)


if __name__ == "__main__":
    main()
