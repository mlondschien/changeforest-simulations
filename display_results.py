from pathlib import Path

import click
import numpy as np
import pandas as pd

_OUTPUT_PATH = Path(__file__).parent.absolute() / "output"


@click.command()
@click.option("--file")
def main(file):
    df = pd.concat([pd.read_csv(f) for f in _OUTPUT_PATH.glob(f"{file}_*.csv")])

    print(
        f"""mean score
{df.pivot_table(index="method", columns="dataset", values="score")}

median score
{df.pivot_table(index="method", columns="dataset", values="score", aggfunc=np.median)}

mean n_cpts
{df.pivot_table(index="method", columns="dataset", values="n_cpts")}

mean left hausdorff
{df.pivot_table(index="method", columns="dataset", values="left_hausdorff")}

mean right hausdorff
{df.pivot_table(index="method",	columns="dataset", values="right_hausdorff")}

mean symmetric hausdorff
{df.pivot_table(index="method", columns="dataset", values="symmetric_hausdorff")}

median symmetric hausdorff
{df.pivot_table(index="method",	columns="dataset", values="symmetric_hausdorff", aggfunc=np.median)}

mean time
{df.pivot_table(index="method", columns="dataset", values="time")}

n
{df.pivot_table(index="method", columns="dataset", values="seed", aggfunc=len)}
"""
    )


if __name__ == "__main__":
    main()
