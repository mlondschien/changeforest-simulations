import numpy as np
import pytest

from changeforest_simulations import simulate
from changeforest_simulations._load import load_iris, load_letters
from changeforest_simulations._simulate import simulate_from_data


@pytest.mark.parametrize(
    "scenario, expected_changepoints, expected_shape",
    [
        ("iris", [0, 50, 100, 150], (150, 4)),
        (
            "dirichlet",
            [0, 100, 130, 220, 320, 370, 520, 620, 740, 790, 870, 1000],
            (1000, 20),
        ),
        ("glass", [0, 17, 46, 55, 68, 144, 214], (214, 8)),
        ("wine", [0, 1079, 1272, 1488, 4324, 6462], (6462, 12)),
        ("breast-cancer", [0, 458, 699], (699, 9)),
        (
            "abalone",
            [0, 568, 635, 1122, 1225, 1914, 2305, 2508, 2775]
            + [2817, 2875, 3134, 3249, 3306, 3432, 4066],
            (4066, 9),
        ),
        ("dry-beans", [0, 2027, 3657, 5585, 6107, 7429, 10975, 13611], (13611, 16)),
        ("covertype", [0, 20510, 56264, 65757, 83124, 366425, 578265], (578265, 54)),
        ("change_in_mean", [0, 200, 400, 600], (600, 5)),
        ("change_in_covariance", [0, 200, 400, 600], (600, 5)),
        ("repeated-covertype", None, (100000, 54)),
        ("repeated-dry-beans", None, (5000, 16)),
        ("repeated-wine", None, (5000, 12)),
        ("wine-noise", None, (10000, 12)),
        ("wine-noise__n_observations=100", None, (100, 12)),
        ("iris-no-change", [0, 50], (50, 4)),
        ("glass-no-change", [0, 76], (76, 8)),
        ("wine-no-change", [0, 2836], (2836, 12)),
        ("breast-cancer-no-change", [0, 458], (458, 9)),
        ("abalone-no-change", [0, 689], (689, 9)),
        ("dry-beans-no-change", [0, 3546], (3546, 16)),
        ("dirichlet-no-change", [0, 150], (150, 20)),
        ("change_in_mean-no-change", [0, 200], (200, 5)),
        ("change_in_covariance-no-change", [0, 200], (200, 5)),
        ("dirichlet__n_observations=100__n_segments=10", None, (100, 20)),
    ],
)
def test_simulate(scenario, expected_changepoints, expected_shape):
    changepoints, time_series = simulate(scenario)

    if expected_changepoints is not None:
        np.testing.assert_array_equal(changepoints, expected_changepoints)

    assert time_series.shape == expected_shape


@pytest.mark.parametrize(
    "load, segment_sizes",
    [(load_iris, [1, 2, 3, 4]), (load_iris, range(12)), (load_letters, range(100))],
)
def test_simulate_with_segment_sizes(load, segment_sizes):
    data = load()

    _, time_series = simulate_from_data(
        data, segment_sizes=segment_sizes, minimal_relative_segment_length=None
    )

    assert time_series.shape[0] == sum(segment_sizes)
