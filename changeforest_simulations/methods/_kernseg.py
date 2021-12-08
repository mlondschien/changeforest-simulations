from rpy2.robjects import default_converter, numpy2ri, r
from rpy2.robjects.conversion import localconverter


def kernseg(X, minimal_relative_segment_length):
    D_max = int(1 / minimal_relative_segment_length)
    n, p = X.shape

    with localconverter(default_converter + numpy2ri.converter):
        r.assign("X", r.matrix(X, nrow=n, ncol=p))
        r.assign("D_max", D_max)
        return r(
            """
result = KernSeg::KernSeg_MultiD(X, D_max)

D_06 = as.integer(D_max * 0.6)
n = nrow(X)

c1 = sapply(1 : D_max, function(D) Rfast::Lchoose(n - 1, D - 1))
c2 = 1 : D_max
lm_fit = lm(result$J.est[D_06 : D_max] ~ c1[D_06 : D_max] + c2[D_06 : D_max])

J_penalized = result$J.est - 2 * c1 * lm_fit$coefficients[2] - 2 * c2 * lm_fit$coefficients[3]
D_opt = which.min(J_penalized)

c(0, result$t.est[D_opt, 0:D_opt])
"""
        )
