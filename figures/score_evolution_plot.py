from pathlib import Path

import click
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt

from changeforest_simulations.constants import (
    COLOR_CYCLE,
    METHOD_ORDERING,
    METHOD_RENAMING,
)

output_path = Path(__file__).parents[1].absolute() / "score_evolution_output"
figures_path = Path(__file__).parent

plt.rc("axes", prop_cycle=cycler(color=list(COLOR_CYCLE)))
plt.rcParams.update({"font.size": 12})


@click.command()
@click.option("--file", default=None)
def main(file):
    files = []
    files.extend(output_path.glob(f"{file}_dirichlet_*.csv"))
    files.extend(output_path.glob(f"{file}_dry-beans-noise_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], axis=0)

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

    df["dataset"] = df["dataset"].str.split("__").str[0]
    df["method"] = df["method"].replace(METHOD_RENAMING)
    df = df[lambda x: x["method"].isin(METHOD_ORDERING)]

    duplicates = df[
        ["dataset", "method", "seed", "n_segments", "n_observations"]
    ].duplicated()
    if duplicates.any():
        print(f"There were {duplicates.sum()} duplicates:")
        df = df[~duplicates]

    df = (
        df.groupby(["dataset", "method", "n_observations", "n_segments"])
        .apply(
            lambda x: pd.Series(
                {
                    "mean_score": x["score"].mean(),
                    "sd_score": x["score"].std() / np.sqrt(len(x)),
                    "median_score": x["score"].median(),
                    "mean_time": x["time"].mean(),
                    "sd_time": x["time"].std() / np.sqrt(len(x)),
                    "mean_n_cpts": x["n_cpts"].mean(),
                    "n": x["seed"].count(),
                }
            )
        )
        .reset_index()
    )

    segments = [10, 20, 40, 80]
    fig, axes = plt.subplots(ncols=len(segments), nrows=3, figsize=(16, 11))

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
            axes[2, idx].errorbar(
                df_dry_beans["n_observations"],
                df_dry_beans["mean_time"],
                yerr=df_dry_beans["sd_time"],
                label=method,
            )

        axes[0, idx].set_title(f"{n_segments} segments")
        axes[0, idx].set_xscale("log")
        # Expand range between ~0.8 - 1.
        axes[0, idx].set_yscale("function", functions=(np.exp, np.log))

        axes[1, idx].set_xscale("log")
        axes[1, idx].set_yscale("function", functions=(np.exp, np.log))

        axes[2, idx].set_xlabel("n")
        axes[2, idx].set_xscale("log")
        axes[2, idx].set_yscale("log")

    axes[-1, -1].legend(loc="lower right")
    axes[0, 0].set_ylabel("avg. adj. Rand index")
    axes[1, 0].set_ylabel("avg. adj. Rand index")
    axes[2, 0].set_ylabel("time (s)")

    plt.figtext(0.485, 0.97, "dirichlet", {"size": 18})
    plt.figtext(0.48, 0.644, "dry beans", {"size": 18})
    plt.figtext(0.48, 0.326, "dry beans", {"size": 18})
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.28)
    plt.savefig(figures_path / "evolution_by_n_observations.eps", dpi=300)
    plt.savefig(figures_path / "evolution_by_n_observations.png", dpi=300)


if __name__ == "__main__":
    main()
