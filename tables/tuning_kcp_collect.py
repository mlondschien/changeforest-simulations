import logging
from pathlib import Path

import click

from changeforest_simulations import HEADER, benchmark

_OUTPUT_FOLDER = Path(__file__).parents[1].absolute() / "output" / "tuning_kcp"
logger = logging.getLogger(__file__)


@click.command()
@click.option("--file", default=None, help="Filename to use.")
@click.option("--n-seeds", default=100, help="Number of seeds to use for simulation.")
@click.option("--seed-start", default=0, help="Seed from which to start iteration.")
def main(file, n_seeds, seed_start):
    _OUTPUT_FOLDER.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    datasets = [
        "iris",
        "glass",
        "wine",
        "breast-cancer",
        "abalone",
        "dry-beans",
        "change_in_mean",
        "change_in_covariance",
        "dirichlet",
    ]

    for seed in range(seed_start, seed_start + n_seeds):
        file_path = _OUTPUT_FOLDER / f"{file}_{seed}.csv"
        if file_path.exists():
            raise ValueError(f"File {file_path} already exists.")
        else:
            file_path.write_text(HEADER)

        logger.info(f"Writing results to {file_path}.")
        for dataset in datasets:
            for method in [
                f"kernseg_rbf__gamma={gamma}"
                for gamma in [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
            ] + ["kernseg_linear", "kernseg_cosine"]:
                benchmark(method, dataset, seed, file_path=file_path)


if __name__ == "__main__":
    main()
