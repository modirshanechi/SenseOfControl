# Analysis of data of Experiment 2 (+ combined with Experiment 1)

Description of different scripts, to be run in the same order:
1.  `Outlierstats.jl`: Showing the statistics of outliers
2.  `SubBySub_inference.jl` + `SubBySub_cleaning.jl`: Performing non-hierarchical inference
3.  `GroupLevel_Minference.jl`: Performing hierarchical inference for models
4.  `GroupLevel_Linference.jl`: Performing constrained inference for l-values
5.  `GroupLevel_Ainference.jl`: Performing inference for Na model
6.  `GroupLevel_Linference_CV.jl`: Performing non-constrained inference for l-values for cross-validation
7.  `GroupLevel_plot.jl`: Plotting the results of inference separately for the two blocks
8.  `GroupLevel_plot_comparative.jl`: Plotting the results of inference comparing the two blocks
9. `GroupLevel_PPC.jl`: Plotting PPC results
10. `GroupLevel_surveys.jl`: Survey analysis for Experiments 1-2
11. `FeedbackResults.jl`: Control analysis for the effect of the feedback trials for Experiments 1-2
