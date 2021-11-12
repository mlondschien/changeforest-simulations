import pytest

from changeforest_simulations import load


@pytest.mark.parametrize(
    "dataset, expected_shape",
    [
        ("iris", (150, 5)),
        ("letters", (20000, 17)),
        ("red_wine", (1599, 12)),
        ("white_wine", (4898, 12)),
    ],
)
def test_load(dataset, expected_shape):
    data = load(dataset)
    assert data.shape == expected_shape

    # Load twice as we might have downloaded directly from openml the first time.
    data = load(dataset)
    assert data.shape == expected_shape
