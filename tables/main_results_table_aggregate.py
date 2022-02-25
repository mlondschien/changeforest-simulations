# Script to aggregate data from main_results_table_collect to a table.
# Call this script with
# `python main_results_table_aggregate.py`
from pathlib import Path

import click
import numpy as np
import pandas as pd

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output"


@click.command()
@click.option("--file", default=None, help="Filename to use.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
def main(file, datasets, methods):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    if methods is not None:
        methods = methods.split(" ")
        df = df[df["method"].isin(methods)]

    if datasets is not None:
        datasets = datasets.split(" ")
        df = df[df["dataset"].isin(datasets)]

    # score
    df_print = df.groupby(["method", "dataset"])["score"].apply(
        lambda x: f"{np.mean(x):.3f} ({np.std(x):.3f})"
    )
    df_print = df_print.reset_index().pivot(index=["method"], columns=["dataset"])
    df_print.columns = df_print.columns.get_level_values(level=1)
    df_print = df_print[datasets]

    print(df_print.to_latex())

    # time
    df_print = df.groupby(["method", "dataset"])["time"].apply(
        lambda x: fmt(np.mean(x))
    )
    df_print = df_print.reset_index().pivot(index=["method"], columns=["dataset"])
    print(df_print)


def fmt(x):
    if np.log10(x) > 1:
        return f"{x:.0f}"
    elif np.log10(x) > 0:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"


if __name__ == "__main__":
    main()

# python tables/simulation_results.py -n1 --datasets 'change_in_mean change_in_covariance dirichlet iris glass wine breast-cancer abalone dry-beans repeated-dry-beans covertype' --methods 'change_in_mean_bs changeforest_bs__random_forest_max_depth=8 changekNN_bs ecp kernseg_rbf multirank'
