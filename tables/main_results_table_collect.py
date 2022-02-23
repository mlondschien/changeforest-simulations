# Code to collect data for the main results table in a csv file.
# Call this script with
# `python tables/main_results_table_collect.py`
import logging
from datetime import datetime
from pathlib import Path

import click

from changeforest_simulations import benchmark

_OUTPUT_FOLDER = Path(__file__).parents[1].absolute() / "output"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
@click.option(
    "--continue", "continue_", is_flag=True, help="Continue from previous run."
)
def main(n_seeds, seed_start, methods, datasets, continue_):

    logging.basicConfig(level=logging.INFO)

    if not continue_:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_path = _OUTPUT_FOLDER / f"{now}.csv"
        file_path.write_text("dataset,seed,method,score,n_cpts,time\n")
        logger.info(f"Writing results to {file_path}.")
    else:
        file_path = sorted(_OUTPUT_FOLDER.glob("202*.csv"))[-1]
        logger.info(f"Continuing {file_path}.")

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
        ]
    else:
        datasets = datasets.split(" ")

    if methods is None:
        methods = [
            "changeforest_bs",
            "changekNN_bs",
            "change_in_mean_bs",
            "ecp",
            "multirank",
            "kernseg_rbf",
        ]
    else:
        methods = methods.split(" ")

    for seed in range(seed_start, seed_start + n_seeds):
        for dataset in datasets:
            for method in methods:
                benchmark(method, dataset, seed, file_path=file_path)


if __name__ == "__main__":
    main()
