from pathlib import Path

import click
import numpy as np
import pandas as pd

output_path = Path(__file__).parents[1].absolute() / "output"
files = output_path.glob("202*.csv")


@click.command()
@click.option("-n", default=1)
@click.option("--methods", default=None)
@click.option("--datasets", default=None)
def main(n, methods, datasets):
    last_file = sorted(files)[-int(n)]
    df = pd.read_csv(last_file)

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
    print(df_print)

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
