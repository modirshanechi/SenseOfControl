# Theoretical analyses

Description of different scripts, to be run in the same order:
1.  `01_ModelRecoveryGeneral_data.jl`: general model recovery for the four main models (Random, N-Act, Emp-l, General)
2.  `02_ModelRecoveryGeneral_plot.jl`: plotting the results of `01_ModelRecoveryGeneral_data.jl`
3.  `03_ModelRecovery_data.jl`: model-recovery and parameter-recovery for the range of $\ell$
4.  `04_ModelRecovery_plot.jl`: plotting the results of `03_ModelRecovery_data.jl`
5.  `05_PowerAnalysis_E3PreReg.jl`: bootstrap power analysis for the pre-registered Experiment 3 (see https://osf.io/rnf8v/), using model-evidence from Experiment 2 (or Experiment 3 post-hoc)
6.  `06_Significance_FDR.jl`: FDR correction pooled across all main-figure hypothesis tests
