# Code for survey analysis

Description of different scripts (order does not matter between them):
1.  `01_Experiments1and2.jl`: survey analysis for Experiments 1 and 2 combined (Fig S15-S17); even though it combines both experiments' data, all figures are saved under `results/Experiment2/Survey/`.
2.  `02_Experiments3.jl`: the same survey-analysis pipeline applied to Experiment 3 alone (Fig S18-S19), saved under `results/Experiment3/Survey/`.

Both scripts require the outputs of `../Inference/` (and the group-level model-selection results, `BMS.jld2`) to already exist.