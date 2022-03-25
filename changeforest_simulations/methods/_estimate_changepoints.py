import numpy as np

from changeforest_simulations.utils import string_to_kwargs

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
from .mnwbs._mnwbs import mnwbs
from .multirank.dynkw import autoDynKWRupt
from .ruptures import kernseg_cosine, kernseg_linear, kernseg_rbf


def estimate_changepoints(X, method, minimal_relative_segment_length, **kwargs):

    # Allow to pass kwargs to changeforest via the method name string.
    # E.g. method="changeforest_bs__random_forest_ntrees=20"
    method, additional_kwargs = string_to_kwargs(method)
    kwargs = {**kwargs, **additional_kwargs}

    if method == "ecp":
        return ecp(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "changeforest_bs":
        if (
            "random_forest_mtry" in additional_kwargs
            and additional_kwargs["random_forest_mtry"] == "sqrt"
        ):
            kwargs["random_forest_mtry"] = int(np.sqrt(X.shape[1]))
        return changeforest_bs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "changeforest_sbs":
        return changeforest_sbs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "changekNN_bs":
        return changekNN_bs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "changekNN_sbs":
        return changekNN_sbs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "change_in_mean_bs":
        return change_in_mean_bs(
            X,
            minimal_relative_segment_length=minimal_relative_segment_length,
            **kwargs,
        )
    elif method == "change_in_mean_sbs":
        return change_in_mean_sbs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "multirank":
        __, cpts = autoDynKWRupt(X.T, Kmax=int(1 / minimal_relative_segment_length))
        return np.append([0], cpts[cpts != 0] + 1)
    elif method == "kernseg":
        return kernseg(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "kernseg_rbf":
        return kernseg_rbf(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "kernseg_linear":
        return kernseg_linear(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "kernseg_cosine":
        return kernseg_cosine(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "kcprs":
        return kcprs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    elif method == "mnwbs":
        return mnwbs(
            X, minimal_relative_segment_length=minimal_relative_segment_length, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}.")
