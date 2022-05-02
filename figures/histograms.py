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
    X_MARKER_KWARGS,
)

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "main"
figures_path = Path(__file__).parent

plt.rc("axes", prop_cycle=cycler(color=list(COLOR_CYCLE)))
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})


@click.command()
@click.option("--file", default=None)
@click.option("--method", default=None)
@click.option("--dataset", default=None)
def main(file, method, dataset):
    alpha, _ = simulate(dataset)

    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")], axis=0)
    df = df[lambda x: x["method"].eq(method) & df["dataset"].eq(dataset)]
    if df.empty:
        raise ValueError(f"No entries for method {method} and dataset {dataset}.")

    changepoints = (
        df["estimated_changepoints"].apply(lambda x: json.loads(x)[1:-1]).sum()
    )

    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2))
    ax.hist(changepoints, bins=1000, color="black", range=(0, alpha[-1]))
    ymin, ymax = ax.get_ylim()
    ax.scatter(alpha[1:-1], [ymax] * (len(alpha) - 2), **X_MARKER_KWARGS)
    plt.savefig(figures_path / f"histogram_{dataset}_{method}", dpi=300)


if __name__ == "__main__":
    main()
