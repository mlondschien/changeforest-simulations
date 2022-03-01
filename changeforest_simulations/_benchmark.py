import logging
from time import perf_counter

import pandas as pd

from changeforest_simulations._simulate import simulate
from changeforest_simulations.methods import estimate_changepoints
from changeforest_simulations.score import adjusted_rand_score, hausdorff_distance
from changeforest_simulations.utils import string_to_kwargs

logger = logging.getLogger(__file__)

HEADER = "dataset,seed,method,score,left_hausdorff,right_hausdorff,symmetric_hausdorff,true_changepoints,estimated_changepoints,n_cpts,time\n"


def benchmark(method, dataset, seed, file_path=None):
    if file_path is not None:
        with open(file_path, "r") as f:
            file_header = f.readline()

        if file_header != HEADER:
            raise ValueError(f"File {file_path} does not have the correct header.")

        existing_results = pd.read_csv(file_path)[["dataset", "seed", "method"]]
        if not existing_results[
            lambda x: x["dataset"].eq(dataset)
            & x["seed"].eq(seed)
            & x["method"].eq(method)
        ].empty:
            logger.info(f"Skipping {seed} {dataset} {method}.")
            return

    logger.info(f"Running {seed} {dataset} {method}.")

    change_points, time_series = simulate(dataset, seed=seed)
    _, dataset_kwargs = string_to_kwargs(dataset)
    _, method_kwargs = string_to_kwargs(method)

    if "minimal_relative_segment_length" in method_kwargs:
        minimal_relative_segment_length = method_kwargs[
            "minimal_relative_segment_length"
        ]
    elif "n_segments" in dataset_kwargs:
        minimal_relative_segment_length = 1 / dataset_kwargs["n_segments"] / 10
    else:
        minimal_relative_segment_length = 0.01

    tic = perf_counter()
    estimate = estimate_changepoints(
        time_series,
        method,
        minimal_relative_segment_length=minimal_relative_segment_length,
    )
    toc = perf_counter()

    score = adjusted_rand_score(change_points, estimate)
    left_hausdorff = hausdorff_distance(change_points, estimate)
    right_hausdorff = hausdorff_distance(estimate, change_points)
    symmetric_hausdorff = max(left_hausdorff, right_hausdorff)

    result = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "score": score,
        "left_hausdorff": left_hausdorff,
        "right_hausdorff": right_hausdorff,
        "symmetric_hausdorff": symmetric_hausdorff,
        "true_changepoints": list(change_points),
        "estimated_changepoints": list(estimate),
        "n_cpts": len(estimate) - 2,
        "time": toc - tic,
    }

    if file_path is not None:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

        existing_results = pd.read_csv(file_path)[["dataset", "seed", "method"]]
        if not existing_results[
            lambda x: x["dataset"].eq(dataset)
            & x["seed"].eq(seed)
            & x["method"].eq(method)
        ].empty:
            raise ValueError(f"Duplicate result {dataset} {seed} {method}.")

        with open(file_path, "a") as f:
            f.write(
                f"""{dataset},{seed},{method},{score},{left_hausdorff},{right_hausdorff},{symmetric_hausdorff},"{str(list(change_points))}","{str(list(estimate))}",{len(estimate) - 2},{toc-tic}\n"""
            )

    return result
