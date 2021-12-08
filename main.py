import logging
from time import perf_counter

import pandas as pd

from changeforest_simulations import adjusted_rand_score, load, simulate
from changeforest_simulations.methods import estimate_changepoints

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

seeds = list(range(1))

datasets = [
    "iris",
    "letters",
    "red_wine",
    "white_wine",
]

result_columns = [
    "dataset",
    "seed",
    "method",
    "score",
    "time",
]
results = pd.DataFrame(columns=result_columns)

for dataset in datasets:
    data = load(dataset)

    for seed in seeds:
        change_points, time_series = simulate(data, seed=seed)
        logger.info(f"True changepoints: {list(change_points)}.")

        for method in [
            "changeforest_bs",
            "changeforest_sbs",
            "changekNN_bs",
            "changekNN_sbs",
            "change_in_mean_bs",
            "change_in_mean_sbs",
            "ecp",
            "multirank",
        ]:
            tic = perf_counter()
            estimate = estimate_changepoints(
                time_series, method, minimal_relative_segment_length=0.02
            )
            toc = perf_counter()

            score = adjusted_rand_score(change_points, estimate)

            results = results.append(
                pd.DataFrame(
                    [[dataset, seed, method, score, toc - tic]], columns=result_columns
                )
            )

            logger.info(
                f"Detection for dataset {dataset} seed {seed} method {method} "
                f"found change points {estimate} in {toc - tic:.2f}s. "
                f"Score={score}."
            )


print(results.groupby(["dataset", "method"])["score", "time"].mean())
