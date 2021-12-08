import numpy as np


def simulate(
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
