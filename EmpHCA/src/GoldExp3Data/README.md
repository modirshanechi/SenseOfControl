# Analysis of data of Experiment 3 (+ combined with Experiment 1 and 2)

Experiment 3 was pre-registered at https://osf.io/rnf8v/

Description of different scripts, to be run in the same order:
1.  `PowerAnalysisPreReg.jl`: Running the power analysis for the pre-regsitration
2.  `Outlierstats.jl`: Showing the statistics of outliers
3.  `SubBySub_inference.jl` + `SubBySub_cleaning.jl`: Performing non-hierarchical inference
4.  `GroupLevel_Minference.jl`: Performing hierarchical inference for models
5.  `GroupLevel_Linference.jl`: Performing constrained inference for l-values
6.  `GroupLevel_Ainference.jl`: Performing inference for Na model
7.  `GroupLevel_Linference_CV.jl`: Performing non-constrained inference for l-values for cross-validation
8.  `GroupLevel_plot.jl`: Plotting the results of inference separately for the two blocks
9.  `GroupLevel_normalizedAccCV.jl`: Evaluating and plotting accuracy rates for Experiments 1-3
10. `GroupLevel_PPC.jl`: Plotting PPC results
11. `GroupLevel_surveys.jl`: Survey analysis
