from pathlib import Path

import click
import numpy as np
import pandas as pd

output_path = Path(__file__).parent.absolute() / "output"
files = output_path.glob("2021-*.csv")


@click.command()
@click.option("-n", default=1)
def main(n):
    last_file = sorted(files)[-int(n)]
    df = pd.read_csv(last_file)

    print("mean score\n")
    print(df.pivot_table(index="method", columns="dataset", values="score"))
    print("\n\nmedian score\n")
    print(
        df.pivot_table(
            index="method", columns="dataset", values="score", aggfunc=np.median
        )
    )
    print("\n\nmean n_cpts\n")
    print(df.pivot_table(index="method", columns="dataset", values="n_cpts"))
    print("\n\nmean time\n")
    print(df.pivot_table(index="method", columns="dataset", values="time"))


if __name__ == "__main__":
    main()
