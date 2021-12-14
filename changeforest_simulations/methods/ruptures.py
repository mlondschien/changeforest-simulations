import numpy as np
import ruptures as rpt
from scipy.special import betaln
from sklearn.linear_model import LinearRegression


def kernseg_rbf(X, minimal_relative_segment_length):
    """Wrapper around kernseg with a radial basis function kernel."""
    return kernseg(X, "rbf", minimal_relative_segment_length)


def kernseg_linear(X, minimal_relative_segment_length):
    """Wrapper around kernseg with a linear kernel."""
    return kernseg(X, "linear", minimal_relative_segment_length)


def kernseg(X, kernel, minimal_relative_segment_length):
    """
    Find change points using the kernel based method presented in [1].

    This is a wrapper around ruptures.KernelCPD algorithm plus model selection as
    described in [1] 3.3.2 (see https://github.com/deepcharles/ruptures/issues/223).

    Note that we use `n_bkps_max = 1 / minimal_relative_segment_length` instead of
    `n / sqrt(log(n))`. We observe better estimation performance and much faster runtime
    (~10x) when using the former.

    [1] S. Arlot, A. Celisse, Z. Harchaoui. A Kernel Multiple Change-point Algorithm
        via Model Selection, 2019
    """
    algo = rpt.KernelCPD(kernel=kernel)

    n_bkps_max = int(1 / minimal_relative_segment_length)
    algo.fit(X).predict(n_bkps=n_bkps_max)

    segmentations_values = [[len(X)]] + list(algo.segmentations_dict.values())
    costs = [algo.cost.sum_of_costs(est) for est in segmentations_values]

    # https://stackoverflow.com/questions/21767690/python-log-n-choose-k
    log_nchoosek = [
        -betaln(1 + n_bkps_max - k, 1 + k) - np.log(n_bkps_max + 1)
        for k in range(0, n_bkps_max + 1)
    ]
    X_lm = np.array([log_nchoosek, range(0, n_bkps_max + 1)]).T
    lm = LinearRegression().fit(
        X_lm[int(0.6 * n_bkps_max) :, :], costs[int(0.6 * n_bkps_max) :]
    )
    adjusted_costs = costs - 2 * X_lm.dot(lm.coef_)

    return [0] + segmentations_values[np.argmin(adjusted_costs)]
