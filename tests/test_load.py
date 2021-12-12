import pytest

from changeforest_simulations import load


@pytest.mark.parametrize(
    "dataset, expected_shape",
    [
        ("iris", (150, 5)),
        ("letters", (20000, 17)),
        ("red_wine", (1599, 12)),
        ("white_wine", (4898, 12)),
        ("wine", (1599 + 4898, 13)),
        ("glass", (214, 9)),
    ],
)
def test_load(dataset, expected_shape):
    data = load(dataset)
    assert data.shape == expected_shape

    assert data.drop(columns="class").to_numpy().dtype == "float"  # no object columns

    # Load twice as we might have downloaded directly from openml the first time.
    data = load(dataset)
    assert data.shape == expected_shape
