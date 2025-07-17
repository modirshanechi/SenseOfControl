################################################################################
# Code for non-hierarchical subject-by-subject inference
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

import StatsPlots

Path_Save = "src/GoldExp1Data/Figures/"
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

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]

# ------------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

for i_sub = subjectIDs
# ------------------------------------------------------------------------------
# Inferring preferences
# ------------------------------------------------------------------------------
df = dataDF[dataDF.subject .== i_sub, :]
df = df[df.timeout .== false, :]

Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
as = df.action .+ 1
model = TuringGoldBasicInf(Xinds, as; N_rooms = N_rooms)
g = HMC(0.01, 50, :θ, :βθ)
chn = sample(model, g, MCMCThreads(), chain_lenght, n_chains)
StatsPlots.plot(chn; legend=true)

chn_df = DataFrame(chn)
filter!(row -> row.iteration > burn_in_lenght, chn_df)
chn_df = chn_df[1:sample_lenght:size(chn_df)[1],:]

# ------------------------------------------------------------------------------
# model selection: random, Na, Empl, and General
# ------------------------------------------------------------------------------
Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,1)
Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
modelAll = TuringGoldBasicInfvsEmplvsNa(Xs, Xinds, Nas, as; N_rooms = N_rooms, K = 1)
gAll = Gibbs(HMC(0.01, 50, :θ, :l, :β, :βa, :βθ), MH(:m, :γ))
chnAll = sample(modelAll, gAll,  MCMCThreads(), chain_lenght, n_chains)
StatsPlots.plot(chnAll; legend=true)

chnAll_df = DataFrame(chnAll)
filter!(row -> row.iteration > burn_in_lenght, chnAll_df)
chnAll_df = chnAll_df[1:sample_lenght:size(chnAll_df)[1],:]

# ------------------------------------------------------------------------------
# model selection: l0, l1, l2
# ------------------------------------------------------------------------------
modelemp = TuringGoldMSel(Xs, as, K=1); 
gemp = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:m, :γ))
chnemp = sample(modelemp, gemp,  MCMCThreads(), chain_lenght, n_chains)
StatsPlots.plot(chnemp; legend=true)

chnemp_df = DataFrame(chnemp)
filter!(row -> row.iteration > burn_in_lenght, chnemp_df)
chnemp_df = chnemp_df[1:sample_lenght:size(chnemp_df)[1],:]

# ------------------------------------------------------------------------------
# save results
# ------------------------------------------------------------------------------
save(Path_Save * "inference_data_sub" * string(i_sub) * ".jld2",
        "chn_df",chn_df, "chnAll_df", chnAll_df, "chnemp_df", chnemp_df)
end

