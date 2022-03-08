# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python false_positive_rate_aggregate.py`
from pathlib import Path

import click
import pandas as pd

from changeforest_simulations.constants import DATASET_ORDERING, DATASET_RENAMING
from changeforest_simulations.utils import string_to_kwargs

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output_tuning"


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
        df.groupby(["dataset"] + parameters)["score"].mean().groupby(parameters).mean()
    )
    df_score[("score", "mean")] = df_mean.apply("{:.3f}".format)
    to_latex(df_score)


def to_latex(df):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)
    df = df[[x for x in DATASET_ORDERING if x in df]]
    print(df)
    print(df.to_latex())


if __name__ == "__main__":
    main()
