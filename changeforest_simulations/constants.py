DATASET_RENAMING = {"change_in_mean": "CIM", "change_in_covariance": "CIC"}
DATASET_ORDERING = [
    "CIM",
    "CIC",
    "dirichlet",
    "iris",
    "glass",
    "wine",
    "breast-cancer",
    "abalone",
    "dry-beans",
    "mean",
]

METHOD_RENAMING = {
    "change_in_mean_bs": "change in mean",
    "changeforest_bs": "changeforest (ours)",
    "changekNN_bs": "changekNN",
    "ecp": "ECP",
    "multirank": "MultiRank",
    "kernseg_rbf": "KCP (rbf)",
}
METHOD_ORDERING = [
    "change in mean",
    "changeforest (ours)",
    "changekNN",
    "ECP",
    "KCP (rbf)",
    "MultiRank",
]

# https://personal.sron.nl/~pault/#sec:qualitative
COLORS = {
    "blue": "#4477AA",
    "cyan": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "black": "#000000",
}

COLOR_CYCLE = [
    COLORS["cyan"],  # change in mean
    COLORS["black"],  # ours
    COLORS["grey"],  # changekNN
    COLORS["red"],  # ECP
    COLORS["yellow"],  # KCP (rbf)
    COLORS["green"],  # MultiRank
]
