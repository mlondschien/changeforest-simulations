import numpy as np
import pytest

from changeforest_simulations import (
    adjusted_rand_score,
    hausdorff_distance,
    symmetric_hausdorff_distance,
)


@pytest.mark.parametrize(
    "left, right, expected", [([0.0, 1.0, 2.0], [0, 2], 0), ([0, 1, 2], [0, 1, 2], 1)]
)
def test_adjusted_rand_score(left, right, expected):
    assert adjusted_rand_score(left, right) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([0.0, 1.0, 2.0], [0, 2], 1 / 2),
        ([0, 1, 2], [0, 1, 2], 0),
        ([0, 1, 3], np.array([0, 1, 2, 3]), 0),
    ],
)
def test_hausdorff_distance(left, right, expected):
    assert hausdorff_distance(left, right) == expected


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([0.0, 1.0, 2.0], [0, 2], 1 / 2),
        ([0, 1, 2], [0, 1, 2], 0),
        ([0, 1, 3], np.array([0, 1, 2, 3]), 1 / 3),
    ],
)
def test_symmetric_hausdorff_distance(left, right, expected):
    assert symmetric_hausdorff_distance(left, right) == expected
