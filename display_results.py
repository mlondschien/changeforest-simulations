from pathlib import Path

import click
import numpy as np
import pandas as pd

output_path = Path(__file__).parent.absolute() / "output"
files = output_path.glob("202*.csv")


@click.command()
@click.option("-n", default=1)
def main(n):
    last_file = sorted(files)[-int(n)]
    df = pd.read_csv(last_file)

    print(
        f"""mean score
{df.pivot_table(index="method", columns="dataset", values="score")}

median score
{df.pivot_table(index="method", columns="dataset", values="score", aggfunc=np.median)}

mean n_cpts
{df.pivot_table(index="method", columns="dataset", values="n_cpts")}

mean time
{df.pivot_table(index="method", columns="dataset", values="time")}

n
{df.pivot_table(index="method", columns="dataset", values="seed", aggfunc=len)}
"""
    )


if __name__ == "__main__":
    main()
