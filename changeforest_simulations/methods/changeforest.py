from changeforest import Control, changeforest


def changeforest_bs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "random_forest",
            "bs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )


def changeforest_sbs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "random_forest",
            "sbs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )


def changekNN_bs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "knn",
            "bs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )


def changekNN_sbs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "knn",
            "sbs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )


def change_in_mean_bs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "change_in_mean",
            "bs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )


def change_in_mean_sbs(X, minimal_relative_segment_length):
    return (
        [0]
        + changeforest(
            X,
            "change_in_mean",
            "sbs",
            Control(minimal_relative_segment_length=minimal_relative_segment_length),
        ).split_points()
        + [len(X)]
    )
