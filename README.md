# Simulations

This repository contains source code to reproduce results of [1].

## Installation instructions

Set up a suitable conda environment and install the `changeforest_simulations` package locally.

```bash
conda env create -f environment.yaml
pip install -e .
```

Importantly, results in [1] are based on `changeforest=0.6.0`.

## Figures

The `figures` folder contains Python scripts to reproduce figures in [1].

| Figure | script | output filename |
| ---: | :--- | :---| 
| 1 | `gain_curves.py` | `gain_curves.eps` |
| 2 | `two_step_search.py` | `two_step_search.eps` |
| 3 | `two_step_search_biased.py` | `two_step_search_biased.eps` |
| 4 | `binary_segmentation.py` | `binary_segmentation_abalone_0.eps` |
| 5 | `score_evolution*` (see below) | `evolution_performance.eps` |
| 6 | `score_evolution*` (see below) | `evolution_time.eps` |
| 7 | `histograms.py` | `histograms_dirichlet.eps` |

All scripts produce both an `eps` and a `png` file.

Figures 5, 6 display the result of an extensive simulation study.
For this, the script `score_evolution_collect.py` collects simulation results in `csv` files.
The script is written in such a fashion to be run in a distributed fashion.
For example, to collect simulation results for 500 simulations, as in [1], distributed among 10 machines, run `python figures/score_evolution_collect.py --seed-start 0 --n-seeds 50 --file changeforest`, ..., `python figures/score_evolution_collect.py --seed-start 450 --n-seeds 50 --file changeforest`.
Here, `file` is a name for the `csv` files containing simulation results.
While developing the `changeforest` algorithm, I would use for example `--file changeforest-0.6.0` to later be able to compare results from different versions.

After collecting simulation results in `csv` files, the `score_evolution_collect.py` script will gather these results and produce Figures 5, 6.
For our setting, calling `python figures/score_evolution_collect.py --file changeforest` will produce the figures.

## Tables

The `tables` folder contains Python scripts to reproduce tables in [1].

| Table | script |
| ---: | :--- | 
| 1 | `adj_rand_examples.py` |
| 2 | `main_results_table_*.py` |
| 3 | `main_results_table_*.py` |
| 4 | `false_positive_rate_*.py` |
| 5 | `main_results_table_*.py` |
| 6 | `main_results_table_*.py` |
| 7 | `tuning_*.py` |
| 8 | `tuning_*.py` |
| 9 | `tuning_kcp_*.py` |

All scripts will both print the tables in a display-friendly manner, unless supplied with the argument `--latex`.

All tables except Table 1 display the result of extensive simulation studies.
For this, the `*_collect.py` scripts collect simulation results in `csv` files.
All `*_collect.py` scripts can be supplied with `--file` (simulation name identifier, e.g. `changeforest`), `--seed-start` (e.g. `0`), and `--n-seeds` (e.g. `500`).
This allows distributing workload over multiple nodes.
For example, to collect main simulation results for 500 simulations, as in [1], distributed among 10 machines, run
`python tables/main_results_table_collect.py --file changeforest --seed-start 0 --n-seeds 50`, ..., `python tables/main_results_table_collect.py --file changeforest --seed-start 450 --n-seeds 50`.

After collecting simulation results in `csv` files, the `*_aggregate` scripts, given the same `--file argument`, will gather these results and print the corresponding tables.

## Benchmarking additional change point estimators

This repository is designed to simplify the benchmarking of additional methods for multivariate nonparametric multiple change point detection.
Say you developed a Python package implementing your method.
To run benchmarks for your method
 - Add your package to the `environment.yaml`.
 - Add your method to the `methods` module. For this, add a file to the `changeforest_simulations/methods` folder implementing a function with arguments `X` (2d `numpy.ndarray`), `minimal_relative_segment_length` (float in `(0, 0.5)`) and `**kwargs` (see below for details). The function should return a list or an array with change point estimates, including `0` and `n`.
 Thus, your methods needs to include an automatic model selection procedure. See the `changeforest_simulations/methods/ruptures.py` file for an example. 
 - Add your method to the `estimate_changepoints` function in `changeforest_simulations/methods/_estimate_changepoints.py`.
 - Consider expanding the tests in `tests/test_methods.py` to test your method.
 - Run any of the above commands with `--method <your methodname here>`. E.g., if your method is called `staubsauger`, first run `python tables/main_results_table_collect.py --file staubsauger=0.1.0 --method staubsauger --n-seeds 500` and then `python tables/main_result_table_aggregate.py --file staubsauger=0.1.0`.

 The `**kwargs` argument can be used to parametrize your method.
 If your method can take an additional parameter `voltage`, you can benchmark your method with different parameter configurations by passing those via the name, e.g., `--methods "staubsauger__voltage=120 staubstauger__voltage=230"`.