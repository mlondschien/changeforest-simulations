from pathlib import Path

import numpy as np
from changeforest import changeforest
from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter

_SCRIPT_PATH = Path(__file__).parent / "utils2.R"


def mnwbs(X, minimal_relative_segment_length, **kwargs):

    n, p = X.shape

    r.source(str(_SCRIPT_PATH))

    segments = changeforest(X, "change_in_mean", "wbs").segments[0:100]
    alpha = np.array([s.start for s in segments])
    beta = np.array([s.stop for s in segments])

    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("alpha", r.matrix(alpha, nrow=len(alpha), ncol=1))
        r.assign("beta", r.matrix(beta, nrow=len(alpha), ncol=1))
        r.assign(
            "h",
            5 * (int(1 / minimal_relative_segment_length) * np.log(n) * n) ** (1 / p),
        )

        return [0] + list(r("MNWBS_full(X, X, alpha, beta, h)")) + [n]
