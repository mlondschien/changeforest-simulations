import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as sklearn_adjusted_rand_score


def adjusted_rand_score(true_changepoints, estimated_changepoints):
    """Compute the adjusted rand index between two sets of changepoints.

    Uses sklearn.metrics.adjusted_rand_score under the hood. See their documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    Examples
    --------
    >>> adjusted_rand_score([0, 50, 100, 150], [0, 100, 150])
    0.5681159420289855
    """
    true_changepoints = np.array(true_changepoints, dtype=np.int_)
    estimated_changepoints = np.array(estimated_changepoints, dtype=np.int_)

    y_true = np.zeros(true_changepoints[-1])
    for i, (start, stop) in enumerate(
        zip(true_changepoints[:-1], true_changepoints[1:])
    ):
        y_true[start:stop] = i

    y_estimated = np.zeros(estimated_changepoints[-1])
    for i, (start, stop) in enumerate(
        zip(estimated_changepoints[:-1], estimated_changepoints[1:])
    ):
        y_estimated[start:stop] = i

    return sklearn_adjusted_rand_score(y_true, y_estimated)


def hausdorff_distance(true_changepoints, estimated_changepoints):
    """Compute the Hausdorff distance between two sets of changepoints.

    This is the maximum distance from a true changepoint to its closest estimated
    counterpart."""
    n = true_changepoints[-1]
    distances = cdist(
        np.array(estimated_changepoints).reshape(-1, 1),
        np.array(true_changepoints).reshape(-1, 1),
    )
    return distances.min(axis=0).max() / n


def symmetric_hausdorff_distance(true_changepoints, estimated_changepoints):
    """Compute the symmetric Hausdorff distance between two sets of changepoints."""
    return max(
        hausdorff_distance(true_changepoints, estimated_changepoints),
        hausdorff_distance(estimated_changepoints, true_changepoints),
    )
