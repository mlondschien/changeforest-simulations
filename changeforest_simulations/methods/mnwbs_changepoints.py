import numpy as np
from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter


def mnwbs_changepoints(X, minimal_relative_segment_length):
    n, p = X.shape

    # Same as https://arxiv.org/pdf/1910.13289.pdf, page 12, except for replacing 30
    # with 1 / minimal_relative_segment_length.
    h = 5 * np.power(np.log(n) / (n * minimal_relative_segment_length), 1 / p)
    M = 50

    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("h", h)
        r.assign("M", M)
        r.assign("n", n)
        r.assign("minimal_segment_length", int(minimal_relative_segment_length * n))
        r("set.seed(0)")
        r("intervals <- changepoints::WBS.intervals(M, 1, n)")
        r(
            "bs <- changepoints::WBS.multi.nonpar(t(X), t(X), 1, n, intervals$Alpha, intervals$Beta, h, minimal_segment_length)"
        )
        return [0] + sorted(r("changepoints::tuneBSmultinonpar(bs, t(X))")) + [n]
