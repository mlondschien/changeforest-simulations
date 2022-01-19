import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

import click
import pandas as pd

from changeforest_simulations import adjusted_rand_score, simulate
from changeforest_simulations.methods import estimate_changepoints

_OUTPUT_FOLDER = Path(__file__).parent.absolute() / "output"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
@click.option(
    "--continue", "continue_", is_flag=True, help="Continue from previous run."
)
def benchmark(n_seeds, seed_start, methods, datasets, continue_):

    logging.basicConfig(level=logging.INFO)

    if not continue_:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_path = _OUTPUT_FOLDER / f"{now}.csv"
        file_path.write_text("dataset,seed,method,score,n_cpts,time\n")
        existing_results = pd.DataFrame(columns=["dataset", "seed", "method"])
        logger.info(f"Writing results to {file_path}.")
    else:
        file_path = sorted(_OUTPUT_FOLDER.glob("202*.csv"))[-1]
        existing_results = pd.read_csv(file_path)[["dataset", "seed", "method"]]
        logger.info(f"Continuing {file_path} with {len(existing_results)} results.")

    if datasets is None:
        datasets = [
            "iris",
            "glass",
            "wine",
            "breast-cancer",
            "abalone",
            "dry-beans",
            "covertype",
            "change_in_mean",
            "change_in_covariance",
            "dirichlet",
            "repeated-dry-beans",
            "repeated-covertype",
        ]
    else:
        datasets = datasets.split(" ")

    if methods is None:
        methods = [
            "changeforest_bs",
            # "changeforest_bs__random_forest_n_trees=20",
            "changekNN_bs",
            "change_in_mean_bs",
            "ecp",
            "multirank",
            "kernseg_rbf",
            # "kcprs",
        ]
    else:
        methods = methods.split(" ")

    skip = {
        "repeated_covertype": ["changekNN_bs", "kernseg_rbf", "ecp", "multirank"],
        "letters": ["ecp", "changekNN_bs", "changekNN_sbs", "multirank", "kcprs"],
        "covertype": ["multirank", "ecp", "changekNN_bs", "kernseg_rbf"],
        "dry-beans": ["multirank"],
        "repeated-dry-beans": ["multirank"],
    }

    slow = {
        # "white_wine": ["ecp"],
        # "abalone": ["ecp"],
        # "covertype": ["changeforest_bs"],
        "repeated-dry-beans": ["ecp"],
        "repeated_covertype": ["ecp"],
    }

    minimal_relative_segment_lengths = {
        "repeated-dry-beans": 0.001,
        "repeated-covertype": 0.001,
    }

    for seed in range(seed_start, seed_start + n_seeds):
        for dataset in datasets:
            change_points, time_series = simulate(dataset, seed=seed)

            for method in methods:
                if method in skip.get(dataset, []):
                    continue

                if (
                    any(slow_method in method for slow_method in slow.get(dataset, []))
                    and seed % 10 != 0
                ):
                    continue

                if (dataset, seed, method) in existing_results.itertuples(index=False):
                    logger.info(f"Skipping existing {dataset}, {seed}, {method}.")
                    continue

                logger.info(f"Running {dataset}, {seed}, {method}.")

                tic = perf_counter()
                estimate = estimate_changepoints(
                    time_series,
                    method,
                    minimal_relative_segment_length=minimal_relative_segment_lengths.get(
                        dataset, 0.01
                    ),
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
