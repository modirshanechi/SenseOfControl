# Results structure

One subfolder per experiment: `Experiment1/`, `Experiment2/`, `Experiment3/`. Within each:

- `MCMC_sub<id>.jld2` / `MCMC_sub<id>_CV.jld2`: per-subject inference results, from `../src/Inference/01_SubBySub_inference.jl` and `02_SubBySub_CVinference.jl`.
- `BMS.jld2`: hierarchical Bayesian model selection, from `../src/Inference/04_SubBySub_marginallikelihood.jl`.
- `MCMCDiag/`: convergence-diagnostic figures, from `../src/Inference/03_SubBySub_inference_diag.jl`.
- `ModelSelection/` (with a `ControlBridgeSampling/` sub-subfolder), `Parameters/`, `RoomPref/`, `AccRate/`, `Value/`: figures from the corresponding scripts in `../src/PlottingAnalysis/`.
- `Outlierstats.jl`: a standalone diagnostic script (outlier statistics), not part of the main pipeline; its own output lands alongside it.