from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter


def ecp(X, minimal_relative_segment_length):
    min_size = int(minimal_relative_segment_length * X.shape[0])

    n, p = X.shape
    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("min_size", min_size)
        return r("ecp::e.divisive(X, min.size=min_size)$estimates") - 1
