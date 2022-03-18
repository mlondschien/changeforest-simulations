# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python false_positive_rate_aggregate.py`
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
from changeforest_simulations.utils import string_to_kwargs

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "tuning"


@click.command()
@click.option("--file", default=None, help="Filename to use.")
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    parameters = [
        "random_forest_n_trees",
        "random_forest_max_depth",
        "random_forest_mtry",
    ]
    df[["random_forest_n_trees", "random_forest_max_depth", "random_forest_mtry"]] = df[
        "method"
    ].apply(lambda x: pd.Series(string_to_kwargs(x)[1]).fillna("None"))

    df_score = (
        df.groupby(["dataset"] + parameters)["score"]
        .apply(lambda x: f"{x.mean():.3f} ({x.std():.3f})")
        .reset_index()
        .pivot(index=parameters, columns=["dataset"])
    )
    df_mean = (
        df.groupby(["dataset"] + parameters)[["score"]]
        .apply(lambda x: pd.Series({"mean": x.mean(), "std": x.std()}))
        .groupby(parameters)
        .apply(
            lambda x: f"{x['mean'].mean():.3f} ({np.sqrt(x['std'].pow(2).mean()):.3f})"
        )
    )
    df_score[("score", "mean")] = df_mean
    print("#" * 20 + "\nScore\n" + "# * 20")
    to_latex(df_score, split=True)

    df_time = (
        df.groupby(["dataset"] + parameters)["time"]
        .apply(lambda x: fmt(np.mean(x)))
        .reset_index()
        .pivot(index=parameters, columns=["dataset"])
    )
    print("#" * 20 + "\nTime\n" + "# * 20")
    to_latex(df_time)


def to_latex(df, split=False):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)
    df = df[[x for x in DATASET_ORDERING if x in df]]

    df = df.rename(METHOD_RENAMING)
    df = df.reindex(METHOD_ORDERING, axis=0)

    if split:
        print(df[df.columns[:5]].to_latex())
        print(df[df.columns[5:]].to_latex())

    else:
        print(df.to_latex())


def fmt(x):
    if np.log10(x) > 1:
        return f"{x:.0f}"
    elif np.log10(x) > 0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


if __name__ == "__main__":
    main()
