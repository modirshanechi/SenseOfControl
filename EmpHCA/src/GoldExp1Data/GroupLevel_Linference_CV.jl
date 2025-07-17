################################################################################
# Performing non-constrained inference for l-values for cross-validation
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random
using Turing, MCMCChains, Distributions
using DataFrames
using CSV
using JLD2
using AdvancedMH
using HypothesisTests

import StatsPlots

Path_Save = "src/GoldExp1Data/Figures/CVAccuracy/"
Path_Load_group = "src/GoldExp1Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp1Data/Figures/"
Path_Load = "data/Experiment1/clean/"

n_chains = 3
chain_lenght = 2500
burn_in_lenght = 500
sample_lenght = 10


# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))
surveyDF = DataFrame(CSV.File(Path_Load * "SurveyData.csv"))
metaDF_survey = DataFrame(CSV.File(Path_Load * "SurveyMetaData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]

# ------------------------------------------------------------------------------
# Load group level inference data
# ------------------------------------------------------------------------------
BMSdata = load(Path_Load_group * "BMS.jld2")

# ------------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

subjectIDsL = BMSdata["subjectIDs_L"]

chnemp_dfs = []
chnemp_dfsPlusOthers = []
for i_sub = subjectIDs
    df = dataDF[dataDF.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,1)
    as = df.action .+ 1
    
    Xinds_set1 = typeof(Xinds)([]); Xinds_set2 = typeof(Xinds)([]);
    Xs_set1 = typeof(Xs)([]);       Xs_set2 = typeof(Xs)([]);
    as_set1 = typeof(as)([]);       as_set2 = typeof(as)([]); 
    for i = eachindex(Xs)
        x = Xinds[i]; x_tilde = [x[2],x[1]]
        if x_tilde ∈ Xinds
            j = findmax([x_temp == x_tilde for x_temp = Xinds])[2]
            if j>i
                push!(Xinds_set1,Xinds[i]); 
                push!(Xs_set1,Xs[i]); 
                push!(as_set1,as[i]);

                push!(Xinds_set2,Xinds[j]); 
                push!(Xs_set2,Xs[j]); 
                push!(as_set2,as[j]);
                
            end
        end
    end

    modelemp_set1 = TuringGold(Xs_set1, as_set1; K = 1)
    gemp_set1 = Gibbs(HMC(0.01, 50, :l, :β), MH(:γ))
    chnemp_set1 = sample(modelemp_set1, gemp_set1,  MCMCThreads(),
                            chain_lenght, n_chains)
    chnemp_df_set1 = DataFrame(chnemp_set1)
    filter!(row -> row.iteration > burn_in_lenght, chnemp_df_set1)
    chnemp_df_set1 = chnemp_df_set1[1:sample_lenght:size(chnemp_df_set1)[1],:]

    modelemp_set2 = TuringGold(Xs_set2, as_set2; K = 1)
    gemp_set2 = Gibbs(HMC(0.01, 50, :l, :β), MH(:γ))
    chnemp_set2 = sample(modelemp_set2, gemp_set2,  MCMCThreads(),
                            chain_lenght, n_chains)
    chnemp_df_set2 = DataFrame(chnemp_set2)
    filter!(row -> row.iteration > burn_in_lenght, chnemp_df_set2)
    chnemp_df_set2 = chnemp_df_set2[1:sample_lenght:size(chnemp_df_set2)[1],:]

    data_temp = (; chnemp_df = [chnemp_df_set1, chnemp_df_set2],
                        Xs = [Xs_set1,Xs_set2], 
                        as = [as_set1,as_set2],
                        Xinds = [Xinds_set1,Xinds_set2])
    if i_sub ∈ subjectIDsL
        push!(chnemp_dfs,data_temp)
    end
    push!(chnemp_dfsPlusOthers,data_temp)
end


save(Path_Save * "BMSbasedLHat_CV.jld2", 
        "chnemp_dfs", chnemp_dfs, 
        "chnemp_dfsPlusOthers", chnemp_dfsPlusOthers, 
        "LSubjects", BMSdata["LSubjects"],
        "subjectIDs", subjectIDs, "subjectIDsL", subjectIDsL)
