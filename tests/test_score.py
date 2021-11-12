import pytest

from changeforest_simulations import adjusted_rand_score


@pytest.mark.parametrize(
    "left, right, expected", [([0.0, 1.0, 2.0], [0, 2], 0), ([0, 1, 2], [0, 1, 2], 1)]
)
def test_adjusted_rand_score(left, right, expected):
    assert adjusted_rand_score(left, right) == expected
