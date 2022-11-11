import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd

from changeforest_simulations.constants import DATASET_ORDERING, DATASET_RENAMING

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "tuning_kcp"


@click.command()
@click.option("--latex", is_flag=True, help="Output in LaTeX format.")
@click.option("--file", default=None, help="Filename to use.")
def main(file, latex):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    df_score = (
        df.groupby(["dataset", "method"])["score"]
        .apply(lambda x: f"{x.mean():.2f} ({x.std():.2f})")
        .reset_index()
        .pivot(index=["method"], columns=["dataset"])
    )
    df_mean = (
        df.groupby(["dataset", "method"])[["score"]]
        .apply(lambda x: pd.Series({"mean": x.mean(), "std": x.std()}))
        .groupby(["method"])
        .apply(
            lambda x: f"{x['mean'].mean():.2f} ({np.sqrt(x['std'].pow(2).mean()):.2f})"
        )
    )
    df_score[("score", "average")] = df_mean
    to_latex(df_score, split=True, latex=latex)


def to_latex(df, split=False, latex=True):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)
    df = df[[x for x in DATASET_ORDERING if x in df]]

    if latex:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            if split:
                print(df[df.columns[:5]].to_latex())
                print(df[df.columns[5:]].to_latex())

            else:
                print(df.to_latex())
    else:
        with pd.option_context("display.max_rows", None, "display.width", None):
            print(df)


if __name__ == "__main__":
    main()
