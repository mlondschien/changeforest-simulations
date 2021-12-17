import pytest

from changeforest_simulations import estimate_changepoints, simulate


@pytest.mark.parametrize(
    "method",
    [
        "ecp",
        "multirank",
        "changeforest_bs",
        "changeforest_bs__random_forest_ntrees=10",
        "changeforest_sbs",
        "changekNN_bs",
        "changekNN_sbs",
        "change_in_mean_bs",
        "change_in_mean_sbs",
        "kernseg",
        "kernseg_rbf",
        "kernseg_linear",
        "kcprs",
    ],
)
def test_method(method):
    _, X = simulate("iris")
    changepoints = estimate_changepoints(
        X, method=method, minimal_relative_segment_length=0.02
    )
    assert changepoints[0] == 0
    assert changepoints[-1] == 150

    assert len(changepoints) > 2
