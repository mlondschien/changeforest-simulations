import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

from changeforest_simulations import adjusted_rand_score, load, simulate
from changeforest_simulations.methods import estimate_changepoints

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_path = Path(__file__).parent.absolute() / "output" / f"{now}.csv"
file_path.write_text("dataset,seed,method,score,time\n")

seeds = list(range(250))

datasets = [
    "iris",
    # "letters",
    "red_wine",
    "white_wine",
]

for dataset in datasets:
    data = load(dataset)

    for seed in seeds:
        change_points, time_series = simulate(data, seed=seed)

        for method in [
            "changeforest_bs",
            "changekNN_bs",
            "change_in_mean_bs",
            "ecp",
            "multirank",
            "kernseg",
        ]:
            tic = perf_counter()
            estimate = estimate_changepoints(
                time_series, method, minimal_relative_segment_length=0.02
            )
            toc = perf_counter()

            score = adjusted_rand_score(change_points, estimate)
            with open(file_path, "a") as f:
                f.write(f"{dataset},{seed},{method},{score},{toc-tic}\n")
