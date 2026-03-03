################################################################################
# Code for non-hierarchical subject-by-subject inference
################################################################################
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

Path_Save = "src/GoldExp3Data/Figures/SubBySub/"
Path_Load = "data/Experiment3/clean/"

n_chains = 3
chain_lenght = 2500
burn_in_lenght = 500
sample_lenght = 10

n_chains_2 = 6
chain_lenght_2 = 1500

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]
subjectUIDs = ExcDF.ID[ExcDF.task_outliers .== 0]


# ----------------------------------------------------------------------
# Rooms
# ----------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

for i_sub_ind = eachindex(subjectIDs)
i_sub = subjectIDs[i_sub_ind]
ID = subjectUIDs[i_sub_ind]
Path_Save_ID = Path_Save * "inference_data_sub" * ID * ".jld2"
Path_Save_fig = Path_Save * "inference_data_sub" * ID 
@show ID
if !isfile(Path_Save_ID)
        @show "on it!"
        # ----------------------------------------------------------------------
        # model selection: random, θ, Empl
        # ----------------------------------------------------------------------
        df = dataDF[dataDF.ID .== ID, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,1)
        Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
        as = df.action .+ 1
        
        
        # General model selection for all trials 
        modelAll = TuringGoldBasicInfvsEmplvsNa(Xs, Xinds, Nas, as; N_rooms = N_rooms, K = 1)
        gAll = Gibbs(HMC(0.01, 50, :θ, :l, :β, :βa, :βθ), MH(:m, :γ))
        chnAll = sample(modelAll, gAll,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnAll; legend=true),
                        Path_Save_fig * "_all.pdf")

        chnAll_df = DataFrame(chnAll)
        filter!(row -> row.iteration > burn_in_lenght, chnAll_df)
        chnAll_df = chnAll_df[1:sample_lenght:size(chnAll_df)[1],:]

        # ------------------------------------------------------------------------------
        # model selection: l0, l1, l2
        # ------------------------------------------------------------------------------
        modelemp = TuringGoldMSel(Xs, as, K=1); 
        gemp = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:m, :γ))
        chnemp = sample(modelemp, gemp,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnemp; legend=true),
                                Path_Save_fig * "_emp.pdf")

        chnemp_df = DataFrame(chnemp)
        filter!(row -> row.iteration > burn_in_lenght, chnemp_df)
        chnemp_df = chnemp_df[1:sample_lenght:size(chnemp_df)[1],:]
        # ----------------------------------------------------------------------
        # save results
        # ----------------------------------------------------------------------
        save(Path_Save_ID, "chnAll_df", chnAll_df, "chnemp_df", chnemp_df)
end
end
