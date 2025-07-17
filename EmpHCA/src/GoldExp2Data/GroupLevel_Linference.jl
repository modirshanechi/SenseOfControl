################################################################################
# Code for constrained inference for l-values
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

Path_Save = "src/GoldExp2Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp2Data/Figures/"
Path_Load = "data/Experiment2/clean/"

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
BMSdata = load(Path_Save * "BMS.jld2")

# ------------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

subjectIDsL = [BMSdata["subjectIDsLG"],BMSdata["subjectIDsLB"]]
GorB = [1,0]
chnemp_dfsAll = Vector{Vector{DataFrame}}([])
for i_GB = 1:2
    sIDs = subjectIDsL[i_GB]
    chnemp_dfs = Vector{DataFrame}([])
    for i_sub = sIDs
        df = dataDF[dataDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        df = df[df.Gtrials .== GorB[i_GB],:]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,1)
        as = df.action .+ 1
        
        BMStemp = [BMSdata["BMSG"],BMSdata["BMSB"]][i_GB]
        
        m_hat = findmax(BMStemp.exp_M[sIDs .== i_sub,:][:])[2]
        
        modelemp = TuringGoldMSelCondM(Xs, as, m_hat; K = 1)
        gemp = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:γ))
        chnemp = sample(modelemp, gemp,  MCMCThreads(), chain_lenght, n_chains)

        chnemp_df = DataFrame(chnemp)
        filter!(row -> row.iteration > burn_in_lenght, chnemp_df)
        chnemp_df = chnemp_df[1:sample_lenght:size(chnemp_df)[1],:]
        push!(chnemp_dfs,chnemp_df)
    end
    push!(chnemp_dfsAll,chnemp_dfs)
end

save(Path_Save * "BMSbasedLHat.jld2", 
        "chnemp_dfsAll", chnemp_dfsAll, 
        "LSubjects", [BMSdata["LSubjectsG"],BMSdata["LSubjectsB"]], 
        "subjectIDs", subjectIDs, "subjectIDsL", subjectIDsL)
