from datetime import datetime
from pathlib import Path
from time import perf_counter

import click
import pandas as pd

from changeforest_simulations import adjusted_rand_score, load, simulate
from changeforest_simulations.methods import estimate_changepoints


@click.command()
@click.option("--n-seeds", default=10, help="Number of seeds to use for simulation.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
def benchmark(n_seeds, methods, datasets):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = Path(__file__).parent.absolute() / "output" / f"{now}.csv"
    file_path.write_text("dataset,seed,method,score,time\n")

    seeds = list(range(n_seeds))

    if datasets is None:
        datasets = ["iris", "white_wine", "red_wine", "wine"]
    elif isinstance(datasets, list):
        datasets = datasets
    else:
        datasets = [datasets]

    if methods is None:
        methods = [
            "changeforest_bs",
            "changekNN_bs",
            "change_in_mean_bs",
            "changeforest_sbs",
            "changekNN_sbs",
            "change_in_mean_sbs",
            "ecp",
            "multirank",
            "kernseg",
        ]
    elif isinstance(methods, list):
        methods = methods
    else:
        methods = [methods]

    skip = {
        "letters": ["ecp", "changekNN_bs", "changekNN_sbs", "multirank"],
        "white_wine": ["ecp"],
        "wine": ["ecp", "multirank"],
    }

    for seed in seeds:
        for dataset in datasets:
            data = load(dataset)
            change_points, time_series = simulate(data, seed=seed)

            for method in methods:
                if method in skip.get(dataset, []):
                    continue

                tic = perf_counter()
                estimate = estimate_changepoints(
                    time_series, method, minimal_relative_segment_length=0.02
                )
                toc = perf_counter()

                score = adjusted_rand_score(change_points, estimate)
                with open(file_path, "a") as f:
                    f.write(f"{dataset},{seed},{method},{score},{toc-tic}\n")

    print(pd.read_csv(file_path).groupby(["dataset", "method"]).mean())


if __name__ == "__main__":
    benchmark()
