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
            "letters",
            [0, 752, 1544, 2319, 3092, 3826, 4629, 5363, 6150, 6908, 7644, 8440]
            + [9187, 9970, 10783, 11531, 12299, 13052, 13818, 14604, 15393, 16148]
            + [16912, 17651, 18434, 19239, 20000],
            (20000, 16),
        ),
        (
            "dirichlet",
            [0, 100, 130, 220, 320, 370, 520, 620, 740, 790, 870, 1000],
            (1000, 20),
        ),
        ("change_in_mean", [0, 200, 400, 600], (600, 5)),
        ("change_in_covariance", [0, 200, 400, 600], (600, 5)),
        ("repeated_covertype", None, (100000, 54)),
        ("repeated-dry-beans", None, (5000, 16)),
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
