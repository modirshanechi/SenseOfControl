# Control analysis: bridge-sampling log-evidence vs. BIC-based log-evidence

Description of different scripts, to be run in the same order:
1.  `01_SubBySub_marginallikelihood.jl`: computes the same subject x model log-evidence matrix as `../04_SubBySub_marginallikelihood.jl`, but via bridge sampling on the already-fitted chains, as a validation check on the BIC-based estimate; saves `BridgeSampLmatrix.jld2` per experiment.
2.  `02_SubBySub_marginallikelihood_compare.jl`: compares the BIC-based and bridge-sampling-based log-evidences and plots their agreement.
