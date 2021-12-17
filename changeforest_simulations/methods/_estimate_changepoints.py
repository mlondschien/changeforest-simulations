import numpy as np

from ._kernseg import kernseg
from .changeforest import (
    change_in_mean_bs,
    change_in_mean_sbs,
    changeforest_bs,
    changeforest_sbs,
    changekNN_bs,
    changekNN_sbs,
)
from .ecp import ecp
from .kcprs import kcprs
from .multirank.dynkw import autoDynKWRupt
from .ruptures import kernseg_linear, kernseg_rbf


def estimate_changepoints(X, method, **kwargs):

    # Allow to pass kwargs to changeforest via the method name string.
    # E.g. method="changeforest_bs__random_forest_ntrees=20"
    if "__" in method:
        args = method.split("__")
        method = args[0]
        for arg in args[1:]:
            key, value = arg.split("=")
            kwargs[key] = float(value)

    if method == "ecp":
        return ecp(X, **kwargs)
    elif method == "changeforest_bs":
        return changeforest_bs(X, **kwargs)
    elif method == "changeforest_sbs":
        return changeforest_sbs(X, **kwargs)
    elif method == "changekNN_bs":
        return changekNN_bs(X, **kwargs)
    elif method == "changekNN_sbs":
        return changekNN_sbs(X, **kwargs)
    elif method == "change_in_mean_bs":
        return change_in_mean_bs(X, **kwargs)
    elif method == "change_in_mean_sbs":
        return change_in_mean_sbs(X, **kwargs)
    elif method == "multirank":
        __, cpts = autoDynKWRupt(X.T)
        return np.append([0], cpts[cpts != 0] + 1)
    elif method == "kernseg":
        return kernseg(X, **kwargs)
    elif method == "kernseg_rbf":
        return kernseg_rbf(X, **kwargs)
    elif method == "kernseg_linear":
        return kernseg_linear(X, **kwargs)
    elif method == "kcprs":
        return kcprs(X, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}.")
