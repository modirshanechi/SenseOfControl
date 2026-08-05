# Code for model and parameter inference

Description of different scripts, to be run in the same order:
1.  `01_SubBySub_inference.jl`: non-hierarchical, per-subject MCMC inference (Bradley-Terry, Emp-l, Na, and Emp-l model selection), parallelized across subjects; saves `MCMC_sub<id>.jld2` per subject.
2.  `02_SubBySub_CVinference.jl`: cross-validated per-subject Emp-l fit using a train/test split; saves `MCMC_sub<id>_CV.jld2` per subject.
3.  `03_SubBySub_inference_diag.jl`: loads the results of scripts 1-2 and plots Rhat/ESS convergence diagnostics per experiment and model.
4.  `04_SubBySub_marginallikelihood.jl`: hierarchical Bayesian model selection using BIC-based evidence for the general model comparison (Random vs. Na vs. Emp-l vs. Bradley-Terry) and MCMC-based evidence for the Emp-l range comparison (l<1 vs. l=1 vs. l>1); saves `BMS.jld2` per experiment.

See `ControlBridgeSampling/` for a bridge-sampling-based validation of the log-evidence estimates used in script 4.
