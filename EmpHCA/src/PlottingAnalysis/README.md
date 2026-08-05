# Code for plotting and analyzing inference results

Description of different scripts (order does not matter between them, but each requires the outputs of `../Inference/` to already exist):
1.  `01_ModelSelection.jl`: plots hierarchical Bayesian model-selection results.
2.  `02_ParameterAnalysis.jl`: plots inferred Emp-l parameters ($β$, $\ell$) split by winning model.
3.  `03_RoomPreferences.jl`: plots empirical vs. model-predicted room preferences per subject group.
4.  `04_CorssValidatedAcc.jl`: plots cross-validated normalized choice-prediction accuracy for each candidate model (Na, Klyubin-emp, Emp-1, Hmax, cross-validated Emp-l), grouped by winning model.
5.  `05_ControlAnalysis.jl`: two control analyses for the GB/BG counterbalancing conditions and the feedback trial influence.
6.  `06_ValueAnalysis.jl`: compares inferred per-room value-rankings between the Emp-l and general models to look for heuristics specific to the 'general participants'.

All scripts loop over the 3 experiments, loading `BMS.jld2` and/or per-subject `MCMC_sub<id>.jld2` / `MCMC_sub<id>_CV.jld2` files produced by `../Inference/`, and save figures under `results/Experiment<n>/<ScriptSubfolder>/`. Gold/Bomb branching (via `expspec.has_gb_split`) only applies to Experiment 2.
