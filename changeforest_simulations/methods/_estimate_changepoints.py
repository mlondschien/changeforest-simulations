from .changeforest import (
    change_in_mean_bs,
    change_in_mean_sbs,
    changeforest_bs,
    changeforest_sbs,
    changekNN_bs,
    changekNN_sbs,
)
from .ecp import ecp
from .multirank.dynkw import autoDynKWRupt


def estimate_changepoints(X, method, **kwargs):
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
        return [0] + cpts[cpts != 0] + 1
    else:
        raise ValueError(f"Unknown method: {method}.")