import numpy as np
import pytest

from changeforest_simulations import load_iris, load_letters, simulate


@pytest.mark.parametrize(
    "load, expected_changepoints",
    [
        (load_iris, [0, 50, 100, 150]),
        (
            load_letters,
            [0, 752, 1544, 2319, 3092, 3826, 4629, 5363, 6150, 6908, 7644, 8440]
            + [9187, 9970, 10783, 11531, 12299, 13052, 13818, 14604, 15393, 16148]
            + [16912, 17651, 18434, 19239, 20000],
        ),
    ],
)
def test_simulate(load, expected_changepoints):
    data = load()

    changepoints, time_series = simulate(data)

    np.testing.assert_array_equal(changepoints, expected_changepoints)
