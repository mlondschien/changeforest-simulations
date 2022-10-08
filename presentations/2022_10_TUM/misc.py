import numpy as np


def get_data(changepoints, means, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=changepoints[-1])

    for (start, stop), mean in zip(zip(changepoints[:-1], changepoints[1:]), means):
        X[start:stop] += mean

    return X


def rss(X, guess):
    left = np.sum((X[0:guess] - np.mean(X[0:guess])) ** 2) if guess > 0 else 0
    right = np.sum((X[guess:] - np.mean(X[guess:])) ** 2)
    return (left + right) / len(X)
