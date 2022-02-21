import numpy as np
import pandas as pd

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
    if "__" in scenario:
        scenario, kwargs = scenario.split("__", 1)
        kwargs = dict(kwargs.split("=") for kwargs in kwargs.split("__"))
        kwargs = {k: int(v) if v.isdigit() else v for k, v in kwargs.items()}
    else:
        kwargs = {}

    if scenario in DATASETS:
        change_points, data = simulate_from_data(load(scenario), seed=seed, **kwargs)
        return change_points, normalize(data)
    elif scenario.endswith("-no-change"):
        changepoints, data = simulate_no_change(scenario, seed=seed)
        return changepoints, normalize(data)
    elif scenario.endswith("-noise"):
        changepoints, data = simulate_with_noise(scenario, seed=seed, **kwargs)
        return changepoints, data
    elif scenario == "repeated-covertype":
        change_points, data = simulate_repeated_covertype(seed=seed)
        return change_points, normalize(data)
    elif scenario == "repeated-dry-beans":
        change_points, data = simulate_repeated_dry_beans(seed=seed)
        return change_points, normalize(data)
    elif scenario == "repeated-wine":
        change_points, data = simulate_repeated_wine(seed=seed)
        return change_points, normalize(data)
    elif scenario == "dirichlet":
        return simulate_dirichlet(seed=seed, **kwargs)
    elif scenario == "change_in_mean":
        return simulate_change_in_mean(seed=seed)
    elif scenario == "change_in_covariance":
        return simulate_change_in_covariance(seed=seed)
    else:
        raise ValueError(f"Scenario {scenario} not supported.")


def simulate_no_change(scenario, seed=0, class_label="class"):
    """Simulate time series without change points by shuffling data.

    Parameters
    ----------
    scenario : str
        One of ...
    seed: int, optional, default=0
        Random seed for reproducibility.
    class_label : str, optional, default="class"
        Column name of class labels in data.
    """
    rng = np.random.default_rng(seed)

    if not scenario.endswith("-no-change"):
        raise ValueError(f"Scenario {scenario} not supported.")

    X = load(scenario[:-10]).drop(columns=class_label).to_numpy()
    rng.shuffle(X)  # This only shuffles along the first axis.
    return np.array([0, len(X)]), X


def simulate_with_noise(
    scenario,
    seed=0,
    class_label="class",
    signal_to_noise=2,
    n_observations=10000,
    n_segments=100,
    minimal_relative_segment_length=None,
):
    if minimal_relative_segment_length is None:
        minimal_relative_segment_length = 1 / n_segments / 10

    rng = np.random.default_rng(seed)

    data = load(scenario[:-6])

    y = data[class_label].to_numpy()

    variances = (  # Compute variances separately for each class, take weighted mean.
        data.groupby(class_label).var() * data.groupby(class_label).count()
    ).sum(axis=0) / (data.shape[0] - data[class_label].nunique())
    X = (data.drop(columns=class_label) / variances.apply(np.sqrt)).to_numpy()

    segment_lengths = _exponential_segment_lengths(
        n_segments, n_observations, minimal_relative_segment_length, seed
    )
    indices = np.array([], dtype="int")

    for _ in range(5):
        try:
            indices = _get_indices(segment_lengths, y, rng, True)
        except ValueError:
            continue

        noise = rng.normal(0, 1 / signal_to_noise, (len(indices), X.shape[1]))

        changepoints = np.append([0], segment_lengths.cumsum())
        time_series = data.iloc[indices].drop(columns=class_label).to_numpy() + noise

        return changepoints, time_series

    raise ValueError("Not enough data")


def _get_indices(segment_lengths, y, rng, replace=True):
    """
    Get indices for segments of lengths `segment_lengths`.

    Parameters
    ----------
    segment_lengths : array-like of int
        For consequtive values `start`, `end` of `segment_lengths`, indices
        `indices[start:stop]` will be unique and correspond to a single value in `y`.
    y : array-like
        Array-like with class labels. For each segment, indices of that segment
        correspond to entries in `y` with the same value.
    rng : np.random.RandomState
        Random number generator.
    replace : bool, optional, default=True
        Whether not to recycle indices for separate segments.
    """
    segment_id = None
    indices = np.array([], dtype="int")
    value_counts = pd.value_counts(y)
    available_indices = np.ones(len(y), dtype=np.bool_)

    for segment_length in segment_lengths:
        available_segments = value_counts[
            lambda x: (x >= segment_length).to_numpy() & (x.index != segment_id)
        ]

        if len(available_segments) == 0:
            raise ValueError("Not enough data.")

        segment_id = rng.choice(available_segments.index, 1)[0]

        new_indices = rng.choice(
            np.flatnonzero((y == segment_id) & available_indices),
            segment_length,
            replace=False,
        )

        if not replace:
            value_counts.loc[segment_id] -= segment_length
            available_indices[new_indices] = False

        indices = np.concatenate([indices, new_indices])

    return indices


def simulate_from_data(
    data,
    class_label="class",
    segment_sizes=None,
    minimal_relative_segment_length=0.01,
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
    minimal_relative_segment_length : float, optional, default=0.01
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

    data = data.reset_index(drop=True)

    value_counts = data[class_label].value_counts()

    if segment_sizes is None:
        if minimal_relative_segment_length is not None:
            value_counts = value_counts[
                lambda x: x / len(data) > minimal_relative_segment_length
            ]

        idx = np.arange(len(value_counts))
        rng.shuffle(idx)
        value_counts = value_counts.iloc[idx]

        indices = np.array([], dtype=np.int_)

        for label, segment_size in value_counts.to_dict().items():
            indices = np.append(
                indices,
                rng.choice(
                    data[lambda x: x[class_label] == label].index,
                    segment_size,
                    replace=False,
                ),
            )

        segment_sizes = value_counts.to_numpy()

        return (
            np.append([0], segment_sizes.cumsum()),
            data.iloc[indices].drop(columns=class_label).to_numpy(),
        )

    else:
        for _ in range(5):
            try:
                indices = _get_indices(segment_sizes, data[class_label].to_numpy(), rng)
            except ValueError:
                continue

            changepoints = np.append([0], np.array(segment_sizes).cumsum())
            time_series = data.iloc[indices].drop(columns=class_label).to_numpy()

            return changepoints, time_series

        raise ValueError("Not enough data")


def simulate_dirichlet(
    seed=0, n_segments=None, n_observations=None, minimal_relative_segment_length=None,
):
    """
    Simulate histogram-valued dataset as described in Scenario 3, 6.1, [1].

    [1] S. Arlot, A. Celisse, Z. Harchaoui. A Kernel Multiple Change-point Algorithm
        via Model Selection, 2019
    """
    if n_segments is not None or n_observations is not None:
        if minimal_relative_segment_length is None:
            minimal_relative_segment_length = 1 / n_segments / 10
        segment_sizes = _exponential_segment_lengths(
            n_segments, n_observations, minimal_relative_segment_length, seed
        )
        changepoints = np.array([0] + segment_sizes.cumsum())
    else:
        changepoints = [0, 100, 130, 220, 320, 370, 520, 620, 740, 790, 870, 1000]

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


def simulate_repeated_covertype(seed=0):
    return simulate_from_data(
        data=load("covertype"),
        segment_sizes=_exponential_segment_lengths(100, 100000, 0.001, seed),
        minimal_relative_segment_length=None,
        seed=seed,
    )


def simulate_repeated_dry_beans(seed=0):
    return simulate_from_data(
        data=load("dry-beans"),
        segment_sizes=_exponential_segment_lengths(100, 5000, 0.001, seed),
        minimal_relative_segment_length=None,
        seed=seed,
    )


def simulate_repeated_wine(seed=0):
    return simulate_from_data(
        data=load("wine"),
        segment_sizes=_exponential_segment_lengths(100, 5000, 0.001, seed),
        minimal_relative_segment_length=None,
        seed=seed,
    )


def _exponential_segment_lengths(
    n_segments, n_observations, minimal_relative_segment_length=0.01, seed=0,
):
    """Exponential segment lengths.

    Parameters
    ----------
    n_segments : int
        Number of segment lengths to be sampled.
    n_observations : int
        Number of observations in simulated time series. The sum of segment lengths
        returned will be equal to this.
    minimal_relative_segment_length : float, optional, default=0.01
        All segments will be at least `n * minimal_relative_segment_length` long.
    seed: int, optional, default=0
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array with segment lengths.

    """
    rng = np.random.default_rng(seed)

    expo = rng.exponential(scale=1, size=n_segments)
    expo = expo * (1 - minimal_relative_segment_length * n_segments) / expo.sum()
    expo = expo + minimal_relative_segment_length
    assert np.abs(expo.sum() - 1) < 1e-12
    assert np.min(expo) >= minimal_relative_segment_length

    return _cascade_round(expo * n_observations)


def _cascade_round(x):
    """Round floats in x to near integer, preserving their sum.

    Inspired by
    https://stackoverflow.com/questions/792460/how-to-round-floats-to-integers-while-preserving-their-sum

    """

    if np.abs(x.sum() - np.round(x.sum())) > 1e-8:
        raise ValueError("Values in x must sum to an integer value.")

    x_rounded = np.zeros(len(x), dtype=np.int_)
    remainder = 0

    for idx in range(len(x)):
        x_rounded[idx] = np.round(x[idx] + remainder)
        remainder += x[idx] - x_rounded[idx]

    assert np.abs(remainder) < 1e-8

    return x_rounded
