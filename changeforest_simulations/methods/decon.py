from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter
import numpy as np

def decon(X, minimal_relative_segment_length):
    n, p = X.shape

    wsize = max(25, int(2 * np.ceil(minimal_relative_segment_length * n)))
    Kmax = min(int(1 / minimal_relative_segment_length), n - 1)

    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("wsize", wsize)
        r.assign("Kmax", Kmax)
        r("set.seed(0)")
        return (
            [0]
            + list(
                r(
                    "kcpRS::kcpRS(data=X, RS_fun=kcpRS::runMean, RS_name='mean', wsize=wsize, nperm=200, Kmax=Kmax, alpha=0.05)$changePoints"
                )
                - 1
            )
            + [n]
        )
