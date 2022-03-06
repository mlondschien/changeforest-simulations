# Code to collect data for the main results table in a csv file.
# Call this script with
# `python tables/main_results_table_collect.py`
import logging
from pathlib import Path

import click

from changeforest_simulations import HEADER, benchmark

_OUTPUT_FOLDER = Path(__file__).parents[1].absolute() / "output_no_change"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
@click.option("--methods", default=None, help="Methods to benchmark. All if None.")
@click.option("--datasets", default=None, help="Datasets to benchmark. All if None.")
@click.option("--file", default=None, help="Filename to use.")
def main(n_seeds, seed_start, methods, datasets, file):
    _OUTPUT_FOLDER.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    if datasets is None:
        datasets = [
            "iris-no-change",
            "glass-no-change",
            "wine-no-change",
            "breast-cancer-no-change",
            "abalone-no-change",
            "dry-beans-no-change",
            "change_in_mean-no-change",
            "change_in_covariance-no-change",
            "dirichlet-no-change",
        ]
    else:
        datasets = datasets.split(" ")

    if methods is None:
        methods = [
            "changeforest_bs",
            "changeforest_bs__model_selection_alpha=0.02",
            "changeforest_bs__model_selection_n_permutations=499",
            "changeforest_bs__model_selection_n_permutations=499__model_selection_alpha=0.02",
            "changekNN_bs",
            "change_in_mean_bs",
            "ecp",
            "multirank",
            "kernseg_rbf",
        ]
    else:
        methods = methods.split(" ")

    for seed in range(seed_start, seed_start + n_seeds):

        file_path = _OUTPUT_FOLDER / f"{file}_{seed}.csv"
        if file_path.exists():
            raise ValueError(f"File {file_path} already exists.")
        else:
            file_path.write_text(HEADER)

        logger.info(f"Writing results to {file_path}.")

        for dataset in datasets:
            for method in methods:
                if method == "multirank" and "dry-beans" in dataset:
                    continue  # Singular Matrix error

                benchmark(method, dataset, seed, file_path=file_path)


if __name__ == "__main__":
    main()