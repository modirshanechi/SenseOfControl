################################################################################
# Code for inference for Na model
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

Path_Save = "src/GoldExp1Data/Figures/GroupLevel/"
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
# Rooms
# ------------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

chnA_dfs = Vector{DataFrame}([])
for i_sub = subjectIDs
    df = dataDF[dataDF.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
    as = df.action .+ 1
    
    modelA = TuringGoldNa(Nas, as)
    gA = HMC(0.01, 50, :βa)
    chnA = sample(modelA, gA,  MCMCThreads(), chain_lenght, n_chains)

    chnA_df = DataFrame(chnA)
    filter!(row -> row.iteration > burn_in_lenght, chnA_df)
    chnA_df = chnA_df[1:sample_lenght:size(chnA_df)[1],:]
    push!(chnA_dfs,chnA_df)
end

save(Path_Save * "BMSbasedNaFit.jld2", 
        "chnA_dfs", chnA_dfs, "subjectIDs", subjectIDs)
