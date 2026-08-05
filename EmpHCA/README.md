# Dependencies

* [Julia](https://julialang.org/) (1.12.3)

# Usage

To install the necessary Julia packages, follow these steps:

1.	Open a Julia terminal, press `]` to enter the package management mode.
2.	In the package management mode, type `activate .`.
3.	In the package management mode, type `instantiate`.

All Julia packages and dependencies will be installed automatically within this environment. On a regular computer, this takes less than 10 minutes.

`SimpleDemo.ipynb` presents a demo for reading and working with the data, along with an example of fitting $\log \ell$ to a participant's data.

# Data and source files

* The cleaned experimental data are saved in `data/Experiment1/clean/`, `data/Experiment2/clean/`, and `data/Experiment3/clean/`
* Code for model and parameter inference is provided in `src/Inference/` (see its `README.md`)
* Code for plotting and analyzing the inference results is provided in `src/PlottingAnalysis/` (see its `README.md`)
* Code for the survey analysis is provided in `src/Surveys/` (see its `README.md`)
* Code for the model/parameter recovery, power analysis, and FDR correction is provided in `src/Theory/` (see its `README.md`)
* All outputs (fitted chains, figures, statistics) are saved under `results/` (see its `README.md`); `src/Theory/`'s outputs are saved under `src/Theory/Figures/` instead
