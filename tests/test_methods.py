import pytest

from changeforest_simulations import load_iris
from changeforest_simulations.methods import estimate_changepoints
from changeforest_simulations.simulate import simulate


@pytest.mark.parametrize(
    "method",
    [
        "ecp",
        "multirank",
        "changeforest_bs",
        "changeforest_sbs",
        "changekNN_bs",
        "changekNN_sbs",
        "change_in_mean_bs",
        "change_in_mean_sbs",
    ],
)
def test_method(method):
    _, X = simulate(load_iris())
    changepoints = estimate_changepoints(
        X, method=method, minimal_relative_segment_length=0.1
    )
    assert len(changepoints) > 0
