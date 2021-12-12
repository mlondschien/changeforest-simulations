from changeforest import Control, changeforest


def changeforest_bs(X, **kwargs):
    return (
        [0]
        + changeforest(X, "random_forest", "bs", Control(**kwargs)).split_points()
        + [len(X)]
    )


def changeforest_sbs(X, **kwargs):
    return (
        [0]
        + changeforest(X, "random_forest", "sbs", Control(**kwargs)).split_points()
        + [len(X)]
    )


def changekNN_bs(X, **kwargs):
    return (
        [0] + changeforest(X, "knn", "bs", Control(**kwargs)).split_points() + [len(X)]
    )


def changekNN_sbs(X, **kwargs):
    return (
        [0] + changeforest(X, "knn", "sbs", Control(**kwargs)).split_points() + [len(X)]
    )


def change_in_mean_bs(X, **kwargs):
    return (
        [0]
        + changeforest(X, "change_in_mean", "bs", Control(**kwargs)).split_points()
        + [len(X)]
    )


def change_in_mean_sbs(X, **kwargs):
    return (
        [0]
        + changeforest(X, "change_in_mean", "sbs", Control(**kwargs)).split_points()
        + [len(X)]
    )
