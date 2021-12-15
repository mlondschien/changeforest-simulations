import numpy as np

from changeforest_simulations import DATASETS, load


def normalize(X):
    """Normalize time series by median pairwise distances."""
    medians = np.median(np.abs(X[1:, :] - X[:-1, :]), axis=0)
    medians[medians == 0] = 1
    return X / medians


def simulate(scenario, seed=0):
    """Simulate time series with change points from scenario.

    Parameters
    ----------
    scenario : str
        One of ...
    seed: int, optional, default=0
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array with changepoints, including zero and `n`.
    numpy.ndarray
        Simulated time series.
    """
    if scenario in DATASETS:
        change_points, data = simulate_from_data(load(scenario), seed=seed)
        return change_points, normalize(data)
    elif scenario == "dirichlet":
        return simulate_dirichlet(seed=seed)
    elif scenario == "change_in_mean":
        return simulate_change_in_mean(seed=seed)
    elif scenario == "change_in_covariance":
        return simulate_change_in_covariance(seed=seed)
    else:
        raise ValueError(f"Scenario {scenario} not supported.")


def simulate_from_data(
    data,
    class_label="class",
    segment_sizes=None,
    minimal_relative_segment_length=0.02,
    seed=0,
):
    """Simulate time series with change points from labeled dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing dataset for (multi-class) classification.
    class_label : str, optional, default="class"
        Column name of class labels in data.
    segment_sizes : list, optional, default=None
        List of sizes of segments to simulate. If `None`, segment sizes correspond to
        value counts of labels in data.
    minimal_relative_segment_length : float, optional, default=0.02
        Minimal relative length of simulated segments. Classes with smaller relative size
        are ignored.
    seed: int, optional, default=0
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array with changepoints, including zero and `n`.
    numpy.ndarray
        Simulated time series.

    """
    rng = np.random.default_rng(seed)

    if segment_sizes is None:
        segment_sizes = data[class_label].value_counts()

        if minimal_relative_segment_length is not None:
            segment_sizes = segment_sizes[
                lambda x: x / len(data) > minimal_relative_segment_length
            ]
    else:
        raise NotImplementedError

    idx = np.arange(len(segment_sizes))
    rng.shuffle(idx)
    segment_sizes = segment_sizes.iloc[idx]

    indices = np.array([], dtype=np.int_)

    for label, segment_size in segment_sizes.to_dict().items():
        indices = np.append(
            indices,
            rng.choice(
                data[lambda x: x[class_label] == label].index,
                segment_size,
                replace=False,
            ),
        )

    return (
        np.append([0], segment_sizes.to_numpy().cumsum()),
        data.iloc[indices].drop(columns=class_label).to_numpy(),
    )


def simulate_dirichlet(seed=0):
    """
    Simulate histogram-valued dataset as described in Scenario 3, 6.1, [1].

    [1] S. Arlot, A. Celisse, Z. Harchaoui. A Kernel Multiple Change-point Algorithm
        via Model Selection, 2019
    """
    d = 20
    changepoints = [0, 100, 130, 220, 320, 370, 520, 620, 740, 790, 870, 1000]
    n_segments = len(changepoints) - 1
    rng = np.random.default_rng(seed)
    params = rng.uniform(0, 0.2, n_segments * d).reshape((n_segments, d))

    X = np.zeros((changepoints[-1], d))
    for idx, (start, end) in enumerate(zip(changepoints[:-1], changepoints[1:])):
        X[start:end, :] = np.random.dirichlet(params[idx, :], end - start)

    return np.array(changepoints), X


def simulate_change_in_mean(seed=0):
    """
    Simulate change in mean dataset as described in [1], 4.3.

    [1] D. Matteson, N. James. A Nonparametric Approach for Multiple Change
    Point Analysis of Multivariate Data, 2012
    """
    T, d, mu = 600, 5, 2

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (T, d))
    X[int(T / 3) : int(2 * T / 3), :] += mu

    return [0, T / 3, 2 * T / 3, T], X


def simulate_change_in_covariance(seed=0):
    """
    Simulate change in correlation dataset as described in [1], 4.3.

    [1] D. Matteson, N. James. A Nonparametric Approach for Multiple Change
    Point Analysis of Multivariate Data, 2012
    """
    T, d, rho = 600, 5, 0.7
    Sigma = np.full((d, d), rho)
    np.fill_diagonal(Sigma, 1)

    rng = np.random.default_rng(seed)
    X = np.concatenate(
        (
            rng.normal(0, 1, (int(T / 3), d)),
            rng.multivariate_normal(np.zeros(d), Sigma, int(T / 3)),
            rng.normal(0, 1, (int(T / 3), d)),
        ),
        axis=0,
    )

    return [0, T / 3, 2 * T / 3, T], X
