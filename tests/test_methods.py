import numpy as np
import pytest

from changeforest_simulations import estimate_changepoints, simulate


@pytest.mark.parametrize(
    "method",
    [
        "ecp",
        "multirank",
        "changeforest_bs",
        "changeforest_bs__random_forest_n_estimators=10",
        "changeforest_sbs",
        "changekNN_bs",
        "changekNN_sbs",
        "change_in_mean_bs",
        "change_in_mean_sbs",
        # "kernseg",
        "kernseg_rbf",
        "kernseg_linear",
        # "kcprs",
        # "mnwbs"  # mnwbs takes 5-10s on iris.
        "rulsif",
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


@pytest.mark.parametrize(
    "method",
    [
        "ecp",
        # "multirank",
        "changeforest_bs",
        "changeforest_bs__random_forest_n_estimators=10",
        "changeforest_sbs",
        "changekNN_bs",
        "changekNN_sbs",
        "change_in_mean_bs",
        "change_in_mean_sbs",
        # "kernseg",
        "kernseg_rbf",
        "kernseg_linear",
        "kernseg_cosine"
        #   "kcprs",
    ],
)
def test_minimal_relative_segment_length(method):
    X = np.zeros((500, 2), dtype=np.float64)
    for i in range(1, 500, 20):
        X[i : (i + 20), :] = i

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, (500, 2))
    X += noise

    few_changepoints = estimate_changepoints(
        X, method, minimal_relative_segment_length=0.2
    )
    assert len(few_changepoints) <= 6

    many_changepoints = estimate_changepoints(
        X, method, minimal_relative_segment_length=0.02
    )
    assert len(many_changepoints) >= 20
