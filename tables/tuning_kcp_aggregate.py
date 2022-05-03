from pathlib import Path

import click
import numpy as np
import pandas as pd

from changeforest_simulations.constants import DATASET_ORDERING, DATASET_RENAMING

_OUTPUT_PATH = Path(__file__).parents[1].absolute() / "output" / "tuning_kcp"


@click.command()
@click.option("--latex", is_flag=True, help="Output in LaTeX format.")
def main(file, latex):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    df_score = (
        df.groupby(["dataset", "method"])["score"]
        .apply(lambda x: f"{x.mean():.3f} ({x.std():.3f})")
        .reset_index()
        .pivot(index=["method"], columns=["dataset"])
    )
    df_mean = (
        df.groupby(["dataset", "method"])[["score"]]
        .apply(lambda x: pd.Series({"mean": x.mean(), "std": x.std()}))
        .groupby(["method"])
        .apply(
            lambda x: f"{x['mean'].mean():.3f} ({np.sqrt(x['std'].pow(2).mean()):.3f})"
        )
    )
    df_score[("score", "average")] = df_mean
    print("\n" + "#" * 50 + "\nScore\n" + "#" * 50 + "\n")
    to_latex(df_score, split=True)


def to_latex(df, split=False, latex=True):
    df.columns = df.columns.get_level_values(level=1)
    df = df.rename(columns=DATASET_RENAMING, copy=False)
    df = df[[x for x in DATASET_ORDERING if x in df]]

    if latex:
        if split:
            print(df[df.columns[:5]].to_latex())
            print(df[df.columns[5:]].to_latex())

        else:
            print(df.to_latex())
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)


if __name__ == "__main__":
    main()
