from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

_DATASET_PATH = Path(__file__).parents[1].resolve() / "datasets"
_LETTERS_PATH = _DATASET_PATH / "letters.csv"
_IRIS_PATH = _DATASET_PATH / "iris.csv"
_WHITE_WINE_PATH = _DATASET_PATH / "winequality-white.csv"
_RED_WINE_PATH = _DATASET_PATH / "winequality-red.csv"


def load_letters():
    if _LETTERS_PATH.exists():
        return pd.read_csv(_LETTERS_PATH)
    else:
        dataset = fetch_openml(data_id=6)["frame"]
        dataset.to_csv(_LETTERS_PATH, index=False)
        return dataset


def load_iris():
    if _IRIS_PATH.exists():
        return pd.read_csv(_IRIS_PATH)
    else:
        dataset = fetch_openml(data_id=61)["frame"]
        dataset.to_csv(_IRIS_PATH, index=False)
        return dataset


def load_red_wine():
    if _RED_WINE_PATH.exists():
        return pd.read_csv(_RED_WINE_PATH)
    else:
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            sep=";",
        )
        dataset = dataset.rename(columns={"quality": "class"}, copy=False)
        dataset.to_csv(_RED_WINE_PATH, index=False)
        return dataset


def load_white_wine():
    if _WHITE_WINE_PATH.exists():
        return pd.read_csv(_WHITE_WINE_PATH)
    else:
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            sep=";",
        )
        dataset = dataset.rename(columns={"quality": "class"}, copy=False)
        dataset.to_csv(_WHITE_WINE_PATH, index=False)
        return dataset


def load_wine():
    white_wine = load_white_wine()
    red_wine = load_red_wine()
    return pd.concat([white_wine.assign(color_red=0), red_wine.assign(color_red=1)])


def load(dataset):
    if dataset == "iris":
        return load_iris()
    elif dataset == "letters":
        return load_letters()
    elif dataset == "red_wine":
        return load_red_wine()
    elif dataset == "white_wine":
        return load_white_wine()
    elif dataset == "wine":
        return load_wine()
    else:
        raise ValueError("Invalid dataset name")
