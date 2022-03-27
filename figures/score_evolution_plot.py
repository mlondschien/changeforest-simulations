from pathlib import Path

import click
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt

from changeforest_simulations.constants import (
    COLOR_CYCLE,
    FIGURE_FONT_SIZE,
    FIGURE_WIDTH,
    METHOD_ORDERING,
    METHOD_RENAMING,
)

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "score_evolution"
figures_path = Path(__file__).parent

plt.rc("axes", prop_cycle=cycler(color=list(COLOR_CYCLE)))
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})


@click.command()
@click.option("--file", default=None)
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")], axis=0)

    df["n_segments"] = (
        df["dataset"]
        .str.split("__")
        .apply(lambda x: dict([y.split("=") for y in x[1:]])["n_segments"])
        .astype(int)
    )

    df["n_observations"] = (
        df["dataset"]
        .str.split("__")
        .apply(lambda x: dict([y.split("=") for y in x[1:]])["n_observations"])
        .astype(int)
    )

    df = df[lambda x: x["n_observations"] <= 64000]

    df["dataset"] = df["dataset"].str.split("__").str[0]
    df["method"] = df["method"].replace(METHOD_RENAMING)
    df = df[lambda x: x["method"].isin(METHOD_ORDERING)]

    index_columns = ["dataset", "method", "n_segments", "n_observations"]
    if df[index_columns + ["seed"]].duplicated().any():
        raise ValueError("There were duplicates.")

    if not df.groupby(index_columns).size().eq(500).all():
        raise ValueError("Not 500 unique seeds per combination.")

    df = (
        df.groupby(["dataset", "method", "n_observations", "n_segments"])
        .apply(
            lambda x: pd.Series(
                {
                    "mean_score": x["score"].mean(),
                    "sd_score": 2 * x["score"].std() / np.sqrt(len(x)),
                    "median_score": x["score"].median(),
                    "mean_time": x["time"].mean(),
                    "sd_time": 2 * x["time"].std() / np.sqrt(len(x)),
                    "mean_n_cpts": x["n_cpts"].mean(),
                    "n": x["seed"].count(),
                }
            )
        )
        .reset_index()
    )

    segments = [20, 80]
    figsize = (FIGURE_WIDTH, FIGURE_WIDTH * 2 / 3)

    _, axes = plt.subplots(ncols=len(segments), nrows=2, figsize=figsize)

    for idx, n_segments in enumerate(segments):
        df_plot = df[df["n_segments"].eq(n_segments)].sort_values("n_observations")

        for method in METHOD_ORDERING:
            df_method = df_plot[lambda x: x["method"].eq(method)]
            df_dirichlet = df_method[lambda x: x["dataset"].eq("dirichlet")]
            df_dry_beans = df_method[lambda x: x["dataset"].eq("dry-beans-noise")]

            axes[0, idx].errorbar(
                df_dirichlet["n_observations"],
                df_dirichlet["mean_score"],
                yerr=df_dirichlet["sd_score"],
                label=method,
            )
            axes[1, idx].errorbar(
                df_dry_beans["n_observations"],
                df_dry_beans["mean_score"],
                yerr=df_dry_beans["sd_score"],
                label=method,
            )

        axes[0, idx].set_title(f"{n_segments} segments", {"size": 16})
        axes[0, idx].set_xscale("log")
        # Expand range between ~0.8 - 1.
        axes[0, idx].set_yscale("function", functions=(np.exp, np.log))

        axes[1, idx].set_xscale("log")
        axes[1, idx].set_yscale("function", functions=(np.exp, np.log))
        axes[1, idx].set_xlabel("sample size (n)")

    axes[-1, -1].legend(loc="lower right")
    axes[0, 0].set_ylabel("avg. adj. Rand index")
    axes[1, 0].set_ylabel("avg. adj. Rand index")

    plt.figtext(0.485, 0.97, "dirichlet", {"size": 18})
    plt.figtext(0.48, 0.482, "dry beans", {"size": 18})
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.2)
    plt.savefig(figures_path / "evolution_by_n_observations.eps", dpi=300)
    plt.savefig(figures_path / "evolution_by_n_observations.png", dpi=300)

    figsize = (FIGURE_WIDTH, FIGURE_WIDTH * 0.4)
    _, axes = plt.subplots(ncols=len(segments), nrows=1, figsize=figsize)

    min_time = df.loc[lambda x: x["dataset"].eq("dirichlet"), "mean_time"].min()
    max_time = df.loc[lambda x: x["dataset"].eq("dirichlet"), "mean_time"].max()
    ymin = np.exp(np.log(min_time) - 0.1 * np.log(max_time / min_time))
    ymax = np.exp(np.log(max_time) + 0.1 * np.log(max_time / min_time))

    for idx, n_segments in enumerate(segments):
        df_plot = df[df["n_segments"].eq(n_segments) & df["dataset"].eq("dirichlet")]

        for method in METHOD_ORDERING:
            df_method = df_plot[lambda x: x["method"].eq(method)]

            axes[idx].plot(
                df_method["n_observations"],
                df_method["mean_time"],
                # yerr=df_method["sd_time"],
                label=method,
            )

        axes[idx].set_title(f"{n_segments} segments", {"size": 16})
        axes[idx].set_xscale("log")
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel("sample size (n)")
        axes[idx].set_ylim(ymin, ymax)

    axes[-1].legend(loc="lower right")
    # axes[1].set_yticklabels([])
    axes[0].set_ylabel("time (s)")
    plt.figtext(0.485, 0.95, "dirichlet", {"size": 18})
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(figures_path / "evolution_time.eps", dpi=300)
    plt.savefig(figures_path / "evolution_time.png", dpi=300)

    df_display = (
        df.groupby(["method", "dataset", "n_segments"])
        .apply(
            lambda x: pd.DataFrame(
                {
                    "threshold": [0.8, 0.95],
                    "val": [
                        x.loc[lambda x: x["mean_score"] > thr, "n_observations"].min()
                        / x["n_segments"].min()
                        for thr in [0.8, 0.95]
                    ],
                }
            )
        )
        .reset_index()
    ).drop(columns="level_3")
    print(
        df_display.pivot(
            index=["method", "threshold"], columns=["dataset", "n_segments"]
        )
    )


if __name__ == "__main__":
    main()
