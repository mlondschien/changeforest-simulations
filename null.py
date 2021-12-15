from datetime import datetime
from pathlib import Path

import numpy as np
from changeforest import Control, changeforest

from changeforest_simulations import load

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_path = Path(__file__).parent.absolute() / "output" / f"null_{now}.csv"
file_path.write_text("dataset,seed,method,segmentation,p_value\n")

for dataset in ["iris", "red_wine", "white_wine", "wine"]:
    data = load(dataset)
    X = data.iloc[0:50, :].drop(columns=["class"]).to_numpy()

    for seed in range(1000):

        np.random.default_rng(seed).shuffle(X)

        for segmentation in ["bs", "sbs", "wbs"]:
            for method in ["random_forest", "knn", "random_forest"]:
                estimate = changeforest(
                    X,
                    method,
                    segmentation,
                    Control(minimal_relative_segment_length=0.05),
                )
                with open(file_path, "a") as f:
                    f.write(
                        f"{dataset},{seed},{method},{segmentation}, {estimate.p_value},\n"
                    )
