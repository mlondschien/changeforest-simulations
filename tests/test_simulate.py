import numpy as np
import pytest

from changeforest_simulations import simulate


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
    ],
)
def test_simulate(scenario, expected_changepoints, expected_shape):
    changepoints, time_series = simulate(scenario)

    np.testing.assert_array_equal(changepoints, expected_changepoints)
    assert time_series.shape == expected_shape
