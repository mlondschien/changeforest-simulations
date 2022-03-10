# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python main_results_table_aggregate.py`
from pathlib import Path

import click
import numpy as np
import pandas as pd

from changeforest_simulations.constants import (
    DATASET_ORDERING,
    DATASET_RENAMING,
    METHOD_ORDERING,
    METHOD_RENAMING,
)

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output"


@click.command()
@click.option("--file", default=None, help="Filename to use.")
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    # score
    df_score = df.groupby(["method", "dataset"])["score"].apply(
        lambda x: f"{np.mean(x):.3f} ({np.std(x):.3f})"
    )
    df_score = df_score.reset_index().pivot(index=["method"], columns=["dataset"])
    df_mean = (
        df.groupby(["method", "dataset"])[["score"]]
        .apply(lambda x: pd.Series({"mean": x.mean(), "std": x.std()}))
        .groupby(["method"])
        .apply(
            lambda x: f"{x['mean'].mean():.3f} ({np.sqrt(x['std'].pow(2).mean()):.3f})"
        )
    )
    df_score[("score", "average")] = df_mean
    to_latex(df_score)

    # time
    df_time = df.groupby(["method", "dataset"])["time"].apply(lambda x: fmt(np.mean(x)))
    df_time = df_time.reset_index().pivot(index=["method"], columns=["dataset"])
    to_latex(df_time)

    # n_unique
    print("Comparing n_unique to n. These should be equal!")
    df_n = (
        df.groupby(["method", "dataset"])["seed"]
        .apply(lambda x: f"{len(x)} ({x.nunique()})")
        .reset_index()
        .pivot(index=["method"], columns=["dataset"])
    )
    print(df_n)


def fmt(x):
    if np.log10(x) > 1:
        return f"{x:.0f}"
    elif np.log10(x) > 0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


def to_latex(df):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)
    df = df[[x for x in DATASET_ORDERING if x in df]]

    df = df.rename(METHOD_RENAMING)
    df = df.reindex(METHOD_ORDERING, axis=0)

    print(df.to_latex())


if __name__ == "__main__":
    main()
