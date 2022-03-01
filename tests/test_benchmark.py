import json

import pandas as pd
import pytest

from changeforest_simulations import HEADER, benchmark


@pytest.mark.parametrize("method, dataset", [("change_in_mean_bs", "iris")])
def test_benchmark(method, dataset, tmp_path):
    with open(tmp_path / "benchmark.csv", "w") as f:
        f.write(HEADER)

    result = benchmark(method, dataset, 0, file_path=tmp_path / "benchmark.csv")
    df = pd.read_csv(tmp_path / "benchmark.csv")
    assert len(df) == 1

    df_as_dict = df.to_dict("records")[0]
    df_as_dict["true_changepoints"] = json.loads(df_as_dict["true_changepoints"])
    df_as_dict["estimated_changepoints"] = json.loads(
        df_as_dict["estimated_changepoints"]
    )

    assert result == pytest.approx(df_as_dict)

    _ = benchmark(method, dataset, 0, file_path=tmp_path / "benchmark.csv")
    df = pd.read_csv(tmp_path / "benchmark.csv")
    assert len(df) == 1

    _ = benchmark(method, dataset, 1, file_path=tmp_path / "benchmark.csv")
    assert len(df) == 1
