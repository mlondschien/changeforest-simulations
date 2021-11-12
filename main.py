import logging
from time import perf_counter

from hdcd import Control, hdcd

from changeforest_simulations import adjusted_rand_score, load, simulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

seeds = list(range(5))
datasets = ["iris", "letters"]

# results = pd.DataFrame(columns=["dataset", "seed", "segmentation", "method", "true_changepoints", "estimated_changepoints", "score", "result"])

for dataset in datasets:
    data = load(dataset)

    for seed in seeds:
        change_points, time_series = simulate(data, seed=seed)
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

                logger.info(
                    f"Detection for dataset {dataset} seed {seed} method {method} and segmentation {segmentation} "
                    f"found change points {result.split_points()} in {toc - tic:.2f}s. "
                    f"Score={adjusted_rand_score(change_points, [0] + result.split_points() + [len(time_series)])}."
                )
