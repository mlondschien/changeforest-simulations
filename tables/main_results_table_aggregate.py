# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python main_results_table_aggregate.py`
from pathlib import Path

import click
import numpy as np
import pandas as pd

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output"

DATASET_RENAMING = {"change_in_mean": "CIM", "change_in_covariance": "CIC"}
DATASET_ORDERING = [
    "CIM",
    "CIC",
    "dirichlet",
    "iris",
    "glass",
    "wine",
    "breast-cancer",
    "abalone",
    "dry-beans",
]

METHOD_RENAMING = {
    "change_in_mean_bs": "change in mean",
    "changeforest_bs": "changeforest (ours)",
    "changekNN_bs": "changekNN",
    "ecp": "ECP",
    "multirank": "MultiRank",
    "kernseg_rbf": "KCP (rbf)",
}
METHOD_ORDERING = [
    "change in mean",
    "changeforest (ours)",
    "changekNN",
    "ECP",
    "KCP (rbf)",
    "MultiRank",
]


@click.command()
@click.option("--file", default=None, help="Filename to use.")
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    # score
    df_score = df.groupby(["method", "dataset"])["score"].apply(
        lambda x: f"{np.mean(x):.3f} ({np.std(x):.3f})"
    )
    df_score = df_score.reset_index().pivot(index=["method"], columns=["dataset"])
    to_latex(df_score)

    # time
    df_print = df.groupby(["method", "dataset"])["time"].apply(
        lambda x: fmt(np.mean(x))
    )
    df_print = df_print.reset_index().pivot(index=["method"], columns=["dataset"])
    to_latex(df_print.to_latex())


def fmt(x):
    if np.log10(x) > 1:
        return f"{x:.0f}"
    elif np.log10(x) > 0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


def to_latex(df):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)[DATASET_ORDERING]

    df.index = df.index.rename(METHOD_RENAMING)
    df = df.reindex(METHOD_ORDERING, axis=0)

    print(df.to_latex())


if __name__ == "__main__":
    main()

# python tables/simulation_results.py -n1 --datasets 'change_in_mean change_in_covariance dirichlet iris glass wine breast-cancer abalone dry-beans repeated-dry-beans covertype' --methods 'change_in_mean_bs changeforest_bs__random_forest_max_depth=8 changekNN_bs ecp kernseg_rbf multirank'
