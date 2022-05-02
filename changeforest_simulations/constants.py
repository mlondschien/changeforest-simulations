DATASET_RENAMING = {"change_in_mean": "CIM", "change_in_covariance": "CIC"}
DATASET_ORDERING = [
    "CIM",
    "CIC",
    "Dirichlet",
    "iris",
    "glass",
    "breast-cancer",
    "abalone",
    "wine",
    "dry-beans",
    "average",
]

METHOD_RENAMING = {
    "dirichlet": "Dirichlet",
    "change_in_mean_bs": "change in mean",
    "changeforest_bs": "changeforest",
    "changekNN_bs": "changekNN",
    "ecp": "ECP",
    "multirank": "MultiRank",
    "kernseg_rbf": "KCP",
}
METHOD_ORDERING = [
    "change in mean",
    "changeforest",
    "changekNN",
    "ECP",
    "KCP",
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
    COLORS["purple"],  # changekNN
    COLORS["red"],  # ECP
    COLORS["yellow"],  # KCP (rbf)
    COLORS["green"],  # MultiRank
]

FIGURE_FONT_SIZE = 14
FIGURE_WIDTH = 13
LINEWIDTH = 2.5
DOTTED_LINEWIDTH = 2.5
X_MARKER_KWARGS = {"marker": "x", "color": COLORS["green"], "linewidth": 10, "s": 2}
