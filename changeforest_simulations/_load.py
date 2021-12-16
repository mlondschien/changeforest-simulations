import tempfile
import urllib
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

_DATASET_PATH = Path(__file__).parents[1].resolve() / "datasets"
_LETTERS_PATH = _DATASET_PATH / "letters.csv"
_IRIS_PATH = _DATASET_PATH / "iris.csv"
_WHITE_WINE_PATH = _DATASET_PATH / "winequality-white.csv"
_RED_WINE_PATH = _DATASET_PATH / "winequality-red.csv"
_GLASS_PATH = _DATASET_PATH / "glass.csv"
_EEG_EYE_STATE_PATH = _DATASET_PATH / "eeg_eye_state.csv"
_ABALONE_PATH = _DATASET_PATH / "abalone.csv"
_COVERTYPE_PATH = _DATASET_PATH / "covertype.csv"

DATASETS = [
    "letters",
    "iris",
    "red_wine",
    "white_wine",
    "wine",
    "glass",
    "covertype",
    "abalone",
]


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


def load_glass():
    if _GLASS_PATH.exists():
        return pd.read_csv(_GLASS_PATH)
    else:
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            names=["id", "ri", "na", "mg", "si", "k", "ca", "ba", "fe", "class"],
            sep=",",
        )
        dataset = dataset.drop(columns=["id"])
        dataset.to_csv(_GLASS_PATH, index=False)
        return dataset


def load_eeg_eye_state():
    if _EEG_EYE_STATE_PATH.exists():
        return pd.read_csv(_EEG_EYE_STATE_PATH)
    else:
        temp_eeg_file = Path(tempfile.mkdtemp()) / "eeg_eye_state.arff"
        with urllib.request.urlopen(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        ) as f:
            with open(temp_eeg_file, "wb") as f_out:
                f_out.writelines(f.read().splitlines(True)[20:])

        dataset = pd.read_csv(temp_eeg_file, sep=",", header=None)
        dataset.columns = [f"eeg_{idx}" for idx in range(14)] + ["class"]
        dataset.to_csv(_EEG_EYE_STATE_PATH, index=False)
        return dataset


def load_abalone():
    if _ABALONE_PATH.exists():
        return pd.read_csv(_ABALONE_PATH)
    else:
        dataset = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
            names=[
                "sex",
                "length",
                "diameter",
                "height",
                "whole_weight",
                "shucked_weight",
                "viscera_weight",
                "shell_weight",
                "class",
            ],
        )
        dataset["male"] = dataset["sex"].eq("M").astype("float")
        dataset["infant"] = dataset["sex"].eq("I").astype("float")
        dataset = dataset.drop(columns="sex")
        dataset.to_csv(_ABALONE_PATH, index=False)
        return dataset


def load_covertype():
    if _COVERTYPE_PATH.exists():
        return pd.read_csv(_COVERTYPE_PATH).astype(float)
    else:
        dataset = fetch_openml(data_id=1596)["frame"]
        dataset.to_csv(_COVERTYPE_PATH, index=False)
        return dataset.astype(float)


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
    elif dataset == "glass":
        return load_glass()
    elif dataset == "eeg_eye_state":
        return load_eeg_eye_state()
    elif dataset == "abalone":
        return load_abalone()
    elif dataset == "covertype":
        return load_covertype()
    else:
        raise ValueError(
            f"Invalid dataset name {dataset}. Availabel datasets are {DATASETS}."
        )
