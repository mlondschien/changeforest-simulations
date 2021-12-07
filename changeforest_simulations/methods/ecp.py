from rpy2.robjects import numpy2ri, r


def ecp(X, minimal_relative_segment_length):
    min_size = int(minimal_relative_segment_length * X.shape[0])
    numpy2ri.activate()

    n, p = X.shape
    r.assign("X", r.matrix(X, nrow=n, ncol=p))
    r.assign("min_size", min_size)
    return r("ecp::e.divisive(X, min.size=min_size)$estimates") - 1
