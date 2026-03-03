# Dependencies

* [Julia](https://julialang.org/) (1.12.3)

# Usage

To install the necessary Julia packages, follow these steps:

1.	Open a Julia terminal, press `]` to enter the package management mode.
2.	In the package management mode, type `activate .`.
3.	In the package management mode, type `instantiate`.

All Julia packages and dependencies will be installed automatically within this environment. On a regular computer, this takes less than 10 minutes.

`SimpleDemo.ipynb` presents a demo for reading and working with the data, along with an example of non-hierarchical model comparison.

# Data and source files

* The cleaned experimental data are saved in `data/Experiment1/clean/`, `data/Experiment2/clean/`, and `data/Experiment3/clean/`
* Code for the model and parameter recovery, as well as the FDR correction, is provided in `src/GoldExpTheory/`
* Code for analyzing the experimental data is provided in `src/GoldExp1Data/`, `src/GoldExp2Data/`, and `src/GoldExp3Data/`
