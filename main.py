import logging
from time import perf_counter

import pandas as pd
from hdcd import Control, hdcd

from changeforest_simulations import adjusted_rand_score, load, simulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

seeds = list(range(1))

datasets = [
    ("iris", "class"),
    ("letters", "class"),
    ("red_wine", "quality"),
    ("white_wine", "quality"),
]

result_columns = [
    "dataset",
    "seed",
    "segmentation",
    "method",
    "true_changepoints",
    "estimated_changepoints",
    "score",
    "time",
    "result",
]
results = pd.DataFrame(columns=result_columns)

for dataset, class_label in datasets:
    data = load(dataset)

    for seed in seeds:
        change_points, time_series = simulate(data, seed=seed, class_label=class_label)
        logger.info(f"True changepoints: {list(change_points)}.")

        for segmentation in "bs", "sbs", "wbs":

            for method in "change_in_mean", "random_forest", "knn":
                tic = perf_counter()
                result = hdcd(
                    time_series,
                    method,
                    segmentation,
                    Control(minimal_relative_segment_length=0.01),
                )
                toc = perf_counter()

                estimated_changepoints = result.split_points()
                score = adjusted_rand_score(
                    change_points, [0] + result.split_points() + [len(time_series)]
                )
                time = toc - tic
                results = results.append(
                    pd.DataFrame(
                        [
                            [
                                dataset,
                                seed,
                                segmentation,
                                method,
                                change_points,
                                estimated_changepoints,
                                score,
                                time,
                                result,
                            ]
                        ],
                        columns=result_columns,
                    )
                )

                logger.info(
                    f"Detection for dataset {dataset} seed {seed} method {method} and segmentation {segmentation} "
                    f"found change points {estimated_changepoints} in {time:.2f}s. "
                    f"Score={score}."
                )

print(results.groupby(["dataset", "method", "segmentation"])["score", "time"].mean())
