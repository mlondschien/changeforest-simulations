import logging
from time import perf_counter

from hdcd import Control, hdcd

from changeforest_simulations import adjusted_rand_score, load_letters, simulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

change_points, time_series = simulate(load_letters())

logger.info(f"True changepoints: {change_points}.")

results = {}

for segmentation in "bs", "sbs", "wbs":
    results[segmentation] = {}

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
            f"Detection for {method} and {segmentation} found change points "
            f"{result.split_points()} in {toc - tic:.2f}s. "
            f"Score={adjusted_rand_score(change_points, [0] + result.split_points() + [len(time_series)])}."
        )
        results[segmentation][method] = result
