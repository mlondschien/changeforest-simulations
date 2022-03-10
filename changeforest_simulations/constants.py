DATASET_RENAMING = {"change_in_mean": "CIM", "change_in_covariance": "CIC"}
DATASET_ORDERING = [
    "CIM",
    "CIC",
    "dirichlet",
    "iris",
    "glass",
    "breast-cancer",
    "abalone",
    "wine",
    "dry-beans",
    "average",
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
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
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
