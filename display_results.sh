#!/bin/bash
# Script to display results of simulations.
OUTPUT="$(find output -name '2021-*' | sort | tail -n 1)"
watch "python -c 'import pandas as pd; print(pd.read_csv(\"$OUTPUT\").groupby([\"dataset\", \"method\"]).mean())'"