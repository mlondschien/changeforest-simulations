# Code to collect data for the score evolution plots in a csv file.
# Call this script with
# `python figures/score_evolution_collect.py`
import logging
from pathlib import Path

import click

from changeforest_simulations import HEADER, benchmark

_OUTPUT_FOLDER = Path(__file__).parents[1].absolute() / "output" / "score_evolution"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
@click.option("--file", default=None, help="Filename to use.")
@click.option("--append", is_flag=True, help="Don't raise if csv already exists.")
def main(n_seeds, seed_start, file, append):
    _OUTPUT_FOLDER.mkdir(exist_ok=True)

    method_list = [
        "changeforest_bs",
        "changekNN_bs",
        "change_in_mean_bs",
        "ecp",
        "multirank",
        "kernseg_rbf",
    ]
    n_segments_list = [20, 80]
    n_observations_list = [
        250,
        353,
        500,
        707,
        1000,
        1414,
        2000,
        2828,
        4000,
        5656,
        8000,
        11313,
        16000,
        22627,
        32000,
        45254,
        64000,
        90509,
        128000,
    ]

    dataset_list = [
        "dirichlet",
        "dry-beans-noise",
    ]

    for seed in range(seed_start, seed_start + n_seeds):

        logging.basicConfig(level=logging.INFO)
        file_path = _OUTPUT_FOLDER / f"{file}_{seed}.csv"
        if file_path.exists():
            if not append:
                raise ValueError(f"File {file_path} already exists.")
        else:
            file_path.write_text(HEADER)
        logger.info(f"Writing results to {file_path}.")

        for dataset in dataset_list:
            for n_segments in n_segments_list:
                for n_observations in n_observations_list:
                    dataset_name = f"{dataset}__n_segments={n_segments}__n_observations={n_observations}"
                    for method in method_list:
                        if method == "ecp" and n_observations >= 10000:
                            continue
                        if method == "multirank" and n_observations > 16000:
                            continue
                        if method == "changekNN_bs" and n_observations > 32000:
                            continue
                        if method == "kernseg_rbf" and n_observations > 64000:
                            continue

                        benchmark(method, dataset_name, seed, file_path=file_path)


if __name__ == "__main__":
    main()
