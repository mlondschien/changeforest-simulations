import pytest

from changeforest_simulations import load_iris, load_letters


@pytest.mark.parametrize(
    "load, expected_shape", [(load_iris, (150, 5)), (load_letters, (20000, 17))]
)
def test_load(load, expected_shape):
    data = load()
    assert data.shape == expected_shape

    # Load twice as we might have downloaded directly from openml the first time.
    data = load()
    assert data.shape == expected_shape
