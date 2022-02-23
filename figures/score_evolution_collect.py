# Code to collect data for the score evolution plots in a csv file.
# Call this script with
# `python figures/score_evolution_collect.py`
import logging
from pathlib import Path

import click

from changeforest_simulations import benchmark

_OUTPUT_FOLDER = Path(__file__).parents[1].absolute() / "score_evolution_output"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
@click.option("--file", default=None, help="Filename to use.")
def main(n_seeds, seed_start, file):

    method_list = [
        "changeforest_bs",
        "changeforest_bs__random_forest_n_trees=20",
        "changeforest_bs__random_forest_n_trees=500",
        "changekNN_bs",
        "change_in_mean_bs",
        "ecp",
        "multirank",
        "kernseg_rbf",
    ]
    n_segments_list = [5, 10, 20, 40, 80, 160]
    n_observations_list = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    dataset_list = ["dirichlet", "dry-beans-noise", "breast-cancer-noise", "wine-noise"]

    for seed in range(seed_start, seed_start + n_seeds):

        logging.basicConfig(level=logging.INFO)
        file_path = _OUTPUT_FOLDER / f"{file}_{seed}.csv"
        if file_path.exists():
            raise ValueError(f"File {file_path} already exists.")

        file_path.write_text("dataset,seed,method,score,n_cpts,time\n")
        logger.info(f"Writing results to {file_path}.")

        for dataset in dataset_list:
            for n_segments in n_segments_list:
                for n_observations in n_observations_list:
                    dataset_name = f"{dataset}__n_segments={n_segments}__n_observations={n_observations}"
                    for method in method_list:
                        if method == "ecp" and n_observations >= 32000:
                            continue
                        benchmark(method, dataset_name, seed, file_path=file_path)


if __name__ == "__main__":
    main()
