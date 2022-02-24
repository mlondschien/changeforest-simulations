from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

output_path = Path(__file__).parents[1].absolute() / "score_evolution_output"


@click.command()
@click.option("--dataset", default=None)
@click.option("--file", default=None)
def main(dataset, file):
    if dataset is None:
        raise ValueError("Please provide dataset name via --dataset argument")

    files = output_path.glob(f"{file}_{dataset}_*.csv")
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

    duplicates = df[["method", "n_observations", "seed", "n_segments"]].duplicated()
    if duplicates.any():
        print(f"There were {duplicates.sum()} duplicates:")
        df = df[~duplicates]

    df = (
        df.groupby(["method", "n_observations", "n_segments"])
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

    _, axes = plt.subplots(ncols=4, nrows=2, figsize=(12, 8))

    for idx, n_segments in enumerate(sorted(df["n_segments"].unique())):
        df_plot = df[df["n_segments"] == n_segments].sort_values("n_observations")

        for method in df_plot["method"].unique():
            df_filtered = df_plot[lambda x: x["method"] == method]
            axes[0, idx].errorbar(
                df_filtered["n_observations"],
                df_filtered["mean_score"],
                yerr=df_filtered["sd_score"],
                label=method,
            )
            axes[1, idx].errorbar(
                df_filtered["n_observations"],
                df_filtered["mean_time"],
                yerr=df_filtered["sd_time"],
                label=method,
            )

        axes[0, idx].legend(loc="lower right")
        axes[0, idx].set_title(f"n_segments={n_segments}")
        axes[0, idx].set_xlabel("n")
        axes[0, idx].set_ylabel("score")
        axes[0, idx].set_xscale("log")

        axes[1, idx].set_xlabel("n")
        axes[1, idx].set_ylabel("time")
        axes[1, idx].set_xscale("log")
        axes[1, idx].set_yscale("log")
        axes[1, idx].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"figures/evolution_{dataset}_by_n_observations.png", dpi=300)

    _, axes = plt.subplots(ncols=4, nrows=2, figsize=(12, 8))
    for idx, n_observations in enumerate(sorted(df["n_observations"].unique())[2:6]):
        df_plot = df[df["n_observations"] == n_observations].sort_values("n_segments")

        for method in df_plot["method"].unique():
            df_filtered = df_plot[lambda x: x["method"] == method]
            axes[0, idx].errorbar(
                df_filtered["n_segments"],
                df_filtered["mean_score"],
                yerr=df_filtered["sd_score"],
                label=method,
            )
            axes[1, idx].errorbar(
                df_filtered["n_segments"],
                df_filtered["mean_time"],
                yerr=df_filtered["sd_time"],
                label=method,
            )

        axes[0, idx].legend(loc="lower right")
        axes[0, idx].set_title(f"n_observations={n_observations}")
        axes[0, idx].set_xlabel("n")
        axes[0, idx].set_ylabel("score")
        axes[0, idx].set_xscale("log")

        axes[1, idx].set_xlabel("n")
        axes[1, idx].set_ylabel("time")
        axes[1, idx].set_xscale("log")
        axes[1, idx].set_yscale("log")
        axes[1, idx].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"figures/evolution_{dataset}_by_n_segments.png", dpi=300)

    df["n_by_n_segments_sq"] = df["n_observations"] / df["n_segments"] ** 2
    _, axes = plt.subplots(ncols=4, nrows=2, figsize=(12, 8))
    values = sorted(df["n_by_n_segments_sq"].value_counts().index[0:5])
    for idx, n_by_n_segments_sq in enumerate(values):
        df_plot = df[df["n_by_n_segments_sq"] == n_by_n_segments_sq].sort_values(
            "n_observations"
        )

        for method in df_plot["method"].unique():
            df_filtered = df_plot[lambda x: x["method"] == method]
            axes[0, idx].errorbar(
                df_filtered["n_observations"],
                df_filtered["mean_score"],
                yerr=df_filtered["sd_score"],
                label=method,
            )
            axes[1, idx].errorbar(
                df_filtered["n_observations"],
                df_filtered["mean_time"],
                yerr=df_filtered["sd_time"],
                label=method,
            )

        axes[0, idx].legend(loc="lower right")
        axes[0, idx].set_title(f"n_by_n_segments_sq={n_by_n_segments_sq}")
        axes[0, idx].set_xlabel("n")
        axes[0, idx].set_ylabel("score")
        axes[0, idx].set_xscale("log")

        axes[1, idx].set_xlabel("n")
        axes[1, idx].set_ylabel("time")
        axes[1, idx].set_xscale("log")
        axes[1, idx].set_yscale("log")
        axes[1, idx].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"figures/evolution_{dataset}_by_balanced.png", dpi=300)


if __name__ == "__main__":
    main()
