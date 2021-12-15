from datetime import datetime
from pathlib import Path
from time import perf_counter

import click
import pandas as pd

from changeforest_simulations import adjusted_rand_score, simulate
from changeforest_simulations.methods import estimate_changepoints


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
def benchmark(n_seeds, methods, datasets):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = Path(__file__).parent.absolute() / "output" / f"{now}.csv"

    seeds = list(range(n_seeds))

    if datasets is None:
        datasets = [
            "iris",
            "white_wine",
            "glass",
            "dirichlet",
            "change_in_mean",
            "change_in_covariance",
        ]
    else:
        datasets = datasets.split(" ")

    if methods is None:
        methods = [
            "changeforest_bs__random_forest_ntrees=100",
            "changeforest_bs__random_forest_ntrees=20",
            "changeforest_bs__random_forest_ntrees=500",
            "changekNN_bs",
            "change_in_mean_bs",
            "changeforest_sbs",
            "changekNN_sbs",
            "change_in_mean_sbs",
            "ecp",
            "multirank",
            "kernseg_linear",
            "kernseg_rbf__gamma=0.1",
            "kernseg_rbf__gamma=1",
            "kernseg_rbf__gamma=0.01",
        ]
    else:
        methods = methods.split(" ")

    skip = {
        "letters": ["ecp", "changekNN_bs", "changekNN_sbs", "multirank"],
        "wine": ["multirank"],
    }

    slow = {
        "wine": ["ecp"],
        "white_wine__normalize=True": ["ecp"],
        "white_wine__normalize=False": ["ecp"],
    }

    file_path.write_text("dataset,seed,method,score,n_cpts,time\n")

    for seed in seeds:
        for dataset in datasets:
            change_points, time_series = simulate(dataset, seed=seed)

            for method in methods:
                if method in skip.get(dataset, []) or (
                    method in slow.get(dataset, []) and seed % 5 != 0
                ):
                    continue

                tic = perf_counter()
                estimate = estimate_changepoints(
                    time_series, method, minimal_relative_segment_length=0.02
                )
                toc = perf_counter()

                score = adjusted_rand_score(change_points, estimate)
                with open(file_path, "a") as f:
                    f.write(
                        f"{dataset},{seed},{method},{score},{len(estimate) - 2},{toc-tic}\n"
                    )

    print(pd.read_csv(file_path).groupby(["dataset", "method"]).mean())


if __name__ == "__main__":
    benchmark()
