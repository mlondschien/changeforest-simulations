# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python false_positive_rate_aggregate.py`
from pathlib import Path

import click
import pandas as pd

from changeforest_simulations.constants import (
    DATASET_ORDERING,
    DATASET_RENAMING,
    METHOD_ORDERING,
    METHOD_RENAMING,
)

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output_no_change"


@click.command()
@click.option("--file", default=None, help="Filename to use.")
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])
    df["false_positive"] = df["n_cpts"] > 0
    df = (
        df.groupby(["dataset", "method"])["false_positive"]
        .apply(lambda x: f"{100 * x.mean():.2f}")
        .reset_index()
        .pivot(index="method", columns=["dataset"])
    )
    to_latex(df)


def to_latex(df):
    df.columns = df.columns.get_level_values(level=1)
    renaming = {f"{x}-no-change": f"{y}-no-change" for x, y in DATASET_RENAMING.items()}
    df = df.rename(columns=renaming, copy=False)
    df = df[[f"{x}-no-change" for x in DATASET_ORDERING if f"{x}-no-change" in df]]
    df = df.rename(METHOD_RENAMING)
    df = df.reindex(METHOD_ORDERING, axis=0)
    print(df.to_latex())


if __name__ == "__main__":
    main()