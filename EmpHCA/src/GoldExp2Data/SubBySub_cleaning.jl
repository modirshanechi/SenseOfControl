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

Path_Save_root = "src/GoldExp2Data/Figures/"
Path_Save = Path_Save_root * "SubBySub/"
Path_Load = "data/Experiment2/clean/"

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]
subjectUIDs = ExcDF.ID[ExcDF.task_outliers .== 0]


# ------------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

for i_sub_ind = eachindex(subjectIDs)
        i_sub = subjectIDs[i_sub_ind]
        ID = subjectUIDs[i_sub_ind]
        Path_Load_ID  = Path_Save * "inference_data_sub" * ID * ".jld2"
        Path_Load_IDB = Path_Save * "inference_data_Btrials_sub" * ID * ".jld2"
        @show ID
        temp = load(Path_Load_ID)
        chnAll_df = temp["chnAll_df"]
        chnAll_dfG = temp["chnAll_dfG"]
        chnemp_dfG = temp["chnemp_dfG"]

        temp = load(Path_Load_IDB)
        chnAll_dfB = temp["chnAll_dfB"]
        chnemp_dfB = temp["chnemp_dfB"]

        # ------------------------------------------------------------------------------
        # save results
        # ------------------------------------------------------------------------------
        save(Path_Save_root * "inference_data_sub" * string(i_sub) * ".jld2", 
                "chnAll_df", chnAll_df,
                "chnAll_dfG", chnAll_dfG, "chnemp_dfG", chnemp_dfG,                
                "chnAll_dfB", chnAll_dfB, "chnemp_dfB", chnemp_dfB)

end
