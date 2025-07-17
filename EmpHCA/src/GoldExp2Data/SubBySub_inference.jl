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

Path_Save = "src/GoldExp2Data/Figures/SubBySub/"
Path_Load = "data/Experiment2/clean/"

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
        # model selection: random, Na, Empl, and General
        # ----------------------------------------------------------------------
        df = dataDF[dataDF.ID .== ID, :]
        df = df[df.timeout .== false, :]

        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
        Xs = gold_Room2X_indexbased(Prooms, Xinds, ΔState, ΔStateDict, Xmax, Ymax,1)
        Nas = [[size(Prooms[x[1]])[1],size(Prooms[x[2]])[1]] for x = Xinds]
        as = df.action .+ 1
        
        Ginds = df.Gtrials .== 1
        Binds = df.Gtrials .== 0

        # General model selection for all trials 
        modelAll = TuringGoldBasicInfvsEmplvsNa_E2(Xs, Xinds, Nas, as, Ginds; 
                                                N_rooms = N_rooms, K = 1)
        gAll = Gibbs(HMC(0.01, 50, :θG, :θB, :βθG, :βθB, 
                        :βaG2, :βaB2, :βaG3, :βaB4,
                        :βB3,  :βG4,  :βG5,  :βB5,
                        :lB3,  :lG4,  :lG5,  :lB5), 
                MH(:m, :γB3, :γG4, :γG5, :γB5))
        chnAll = sample(modelAll, gAll,  MCMCThreads(), chain_lenght_2, n_chains_2)
        StatsPlots.savefig(StatsPlots.plot(chnAll; legend=true),
                        Path_Save_fig * "_all.pdf")

        chnAll_df = DataFrame(chnAll)
        filter!(row -> row.iteration > burn_in_lenght, chnAll_df)
        chnAll_df = chnAll_df[1:sample_lenght:size(chnAll_df)[1],:]

        # ----------------------------------------------------------------------
        # Replication for the gold trials
        # ----------------------------------------------------------------------
        modelAllG = TuringGoldBasicInfvsEmplvsNa(Xs[Ginds], Xinds[Ginds], Nas[Ginds], 
                                                as[Ginds]; N_rooms = N_rooms, K = 1)
        gAll = Gibbs(HMC(0.01, 50, :θ, :l, :β, :βa, :βθ), MH(:m, :γ))
        chnAllG = sample(modelAllG, gAll,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnAllG; legend=true),
                        Path_Save_fig * "_allG.pdf")

        chnAll_dfG = DataFrame(chnAllG)
        filter!(row -> row.iteration > burn_in_lenght, chnAll_dfG)
        chnAll_dfG = chnAll_dfG[1:sample_lenght:size(chnAll_dfG)[1],:]
        
        modelempG = TuringGoldMSel(Xs[Ginds], as[Ginds], K=1); 
        gemp = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:m, :γ))
        chnempG = sample(modelempG, gemp,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnempG; legend=true),
                        Path_Save_fig * "_EmpG.pdf")

        chnemp_dfG = DataFrame(chnempG)
        filter!(row -> row.iteration > burn_in_lenght, chnemp_dfG)
        chnemp_dfG = chnemp_dfG[1:sample_lenght:size(chnemp_dfG)[1],:]

        # ----------------------------------------------------------------------
        # Focus on bomb trials
        # ----------------------------------------------------------------------
        modelAllB = TuringGoldBasicInfvsEmplvsNa(Xs[Binds], Xinds[Binds], Nas[Binds], 
                                                as[Binds]; N_rooms = N_rooms, K = 1)
        gAll = Gibbs(HMC(0.01, 50, :θ, :l, :β, :βa, :βθ), MH(:m, :γ))
        chnAllB = sample(modelAllB, gAll,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnAllB; legend=true),
                        Path_Save_fig * "_allB.pdf")

        chnAll_dfB = DataFrame(chnAllB)
        filter!(row -> row.iteration > burn_in_lenght, chnAll_dfB)
        chnAll_dfB = chnAll_dfB[1:sample_lenght:size(chnAll_dfB)[1],:]
        
        modelempB = TuringGoldMSel(Xs[Binds], as[Binds], K=1); 
        gemp = Gibbs(HMC(0.01, 50, :l0, :l2, :β), MH(:m, :γ))
        chnempB = sample(modelempB, gemp,  MCMCThreads(), chain_lenght, n_chains)
        StatsPlots.savefig(StatsPlots.plot(chnempB; legend=true),
                        Path_Save_fig * "_EmpB.pdf")

        chnemp_dfB = DataFrame(chnempB)
        filter!(row -> row.iteration > burn_in_lenght, chnemp_dfB)
        chnemp_dfB = chnemp_dfB[1:sample_lenght:size(chnemp_dfB)[1],:]


        # ----------------------------------------------------------------------
        # save results
        # ----------------------------------------------------------------------
        save(Path_Save_ID, "chnAll_df", chnAll_df, 
                "chnAll_dfG", chnAll_dfG, "chnemp_dfG", chnemp_dfG,
                "chnAll_dfB", chnAll_dfB, "chnemp_dfB", chnemp_dfB)
end
end
