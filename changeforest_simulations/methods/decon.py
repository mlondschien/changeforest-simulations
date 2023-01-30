from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter


def decon(X, minimal_relative_segment_length):
    min_size = int(minimal_relative_segment_length * X.shape[0])
    Kmax = int(1 / minimal_relative_segment_length)

    n, p = X.shape
    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("min_size", min_size)
        r.assign("Kmax", Kmax)
        r("set.seed(0)")
        return (
            [0]
            + list(
                r(
                    "kcpRS::kcpRS(data=X, RS_fun=kcpRS::runMean, RS_name='mean', wsize=2 * min_size, nperm=200, Kmax=Kmax, alpha=0.05)$changePoints"
                )
                - 1
            )
            + [n]
        )
