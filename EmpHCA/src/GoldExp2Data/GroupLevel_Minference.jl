################################################################################
# Code for hierarchical inference for models
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

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExp2Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp2Data/Figures/"
Path_Load = "data/Experiment2/clean/"

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
ExcDF  = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs  = ExcDF.subject[ExcDF.task_outliers .== 0]

# ------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

ProcessedData = []
for i_sub = subjectIDs
    Path_Load_ID = Path_Load_Inf * "inference_data_sub" * 
                                        string(i_sub) * ".jld2"

    @show i_sub
    # ------------------------------------------------------------------------
    # Inference data
    # ------------------------------------------------------------------------
    temp = load(Path_Load_ID)
    chnAll_df = temp["chnAll_df"]
    chnAll_dfG = temp["chnAll_dfG"]
    chnemp_dfG = temp["chnemp_dfG"]
    chnAll_dfB = temp["chnAll_dfB"]
    chnemp_dfB = temp["chnemp_dfB"]
    df = dataDF[dataDF.subject .== i_sub,:]
    
    # ------------------------------------------------------------------------
    # model selection
    # ------------------------------------------------------------------------
    pmAll = [sum(chnAll_df.m .== j) for j = 1:6]; 
    pmAll = pmAll ./ sum(pmAll)
    mAll_hat = findmax(pmAll)[2]

    pmAllG = [sum(chnAll_dfG.m .== j) for j = 1:4]; 
    pmAllG = pmAllG ./ sum(pmAllG)
    mAll_hatG = findmax(pmAllG)[2]

    pmG = [sum(chnemp_dfG.m .== j) for j = 1:3]; pmG = pmG ./ sum(pmG)
    m_hatG = findmax(pmG)[2]

    pmAllB = [sum(chnAll_dfB.m .== j) for j = 1:4]; 
    pmAllB = pmAllB ./ sum(pmAllB)
    mAll_hatB = findmax(pmAllB)[2]

    pmB = [sum(chnemp_dfB.m .== j) for j = 1:3]; pmB = pmB ./ sum(pmB)
    m_hatB = findmax(pmB)[2]
    
    push!(ProcessedData,
        (;  pmAll = pmAll, mAll_hat = mAll_hat,
            pmAllG = pmAllG, mAll_hatG = mAll_hatG,
            pmG = pmG, m_hatG = m_hatG,
            pmAllB = pmAllB, mAll_hatB = mAll_hatB,
            pmB = pmB, m_hatB = m_hatB,
            i_sub = i_sub, Cond = df.GB_condition[1]))
end

# ------------------------------------------------------------------------
# GB-subjects
# ------------------------------------------------------------------------
GBSub_inds  = [d.Cond .== 1 for d = ProcessedData]

# ------------------------------------------------------------------------
# Random effect general model selection
# ------------------------------------------------------------------------
L_matrix = Matrix(hcat([d.pmAll for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); N_model = size(L_matrix)[2]
BMSAll = MCMC_BMS_Statistics(L_matrix; 
                N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)

# ------------------------------------------------------------------------
# Gold trials
# ------------------------------------------------------------------------
L_matrix = Matrix(hcat([d.pmAllG for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); N_model = size(L_matrix)[2]
BMSAllG = MCMC_BMS_Statistics(L_matrix; 
                N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)

L_matrixG = Matrix(hcat([d.pmG for d = ProcessedData]...)')
L_matrixG .= log.(max.(L_matrixG,0.001)); N_model = size(L_matrixG)[2]
mAll_hatG = [findmax(BMSAllG.exp_M[i,:])[2] for i = 1:size(BMSAllG.exp_M)[1]]
LSubjectsG = (1:length(mAll_hatG))[mAll_hatG .== 3]
subjectIDsLG = subjectIDs[LSubjectsG]
L_matrixG = L_matrixG[LSubjectsG,:]
BMSG = MCMC_BMS_Statistics(L_matrixG; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)

# ------------------------------------------------------------------------
# Bomb trials
# ------------------------------------------------------------------------
L_matrix = Matrix(hcat([d.pmAllB for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); N_model = size(L_matrix)[2]
BMSAllB = MCMC_BMS_Statistics(L_matrix; 
                N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)

L_matrixB = Matrix(hcat([d.pmB for d = ProcessedData]...)')
L_matrixB .= log.(max.(L_matrixB,0.001)); N_model = size(L_matrixB)[2]
mAll_hatB = [findmax(BMSAllB.exp_M[i,:])[2] for i = 1:size(BMSAllB.exp_M)[1]]
LSubjectsB = (1:length(mAll_hatB))[mAll_hatB .== 3]
subjectIDsLB = subjectIDs[LSubjectsB]
L_matrixB = L_matrixB[LSubjectsB,:]
BMSB = MCMC_BMS_Statistics(L_matrixB; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)                

# ------------------------------------------------------------------------
# Emp-selection for common L-subjects
# ------------------------------------------------------------------------
LSubjectsAll = (1:length(mAll_hatB))[(mAll_hatG .== 3) .& (mAll_hatB .== 3)]
subjectIDsLAll = subjectIDs[LSubjectsAll]

L_matrixG = Matrix(hcat([d.pmG for d = ProcessedData]...)')
L_matrixG .= log.(max.(L_matrixG,0.001)); N_model = size(L_matrixG)[2]
L_matrixG = L_matrixG[LSubjectsAll,:]
BMSG_commonSubs = MCMC_BMS_Statistics(L_matrixG; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)

L_matrixB = Matrix(hcat([d.pmB for d = ProcessedData]...)')
L_matrixB .= log.(max.(L_matrixB,0.001)); N_model = size(L_matrixB)[2]
L_matrixB = L_matrixB[LSubjectsAll,:]
BMSB_commonSubs = MCMC_BMS_Statistics(L_matrixB; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model)


# ------------------------------------------------------------------------
# Emp-selection for common L-subjects
# ------------------------------------------------------------------------
save(Path_Save * "BMS.jld2", 
        # General model selection
        "BMSAll", BMSAll, "subjectIDs", subjectIDs,
        # Model selection to for gold and bomb blocks separately
        "BMSAllG", BMSAllG, "BMSG", BMSG, "mAll_hatG", mAll_hatG,
        "BMSAllB", BMSAllB, "BMSB", BMSB, "mAll_hatB", mAll_hatB,
        "LSubjectsG", LSubjectsG, "subjectIDsLG", subjectIDsLG,
        "LSubjectsB", LSubjectsB, "subjectIDsLB", subjectIDsLB,
        # Conditional model selection to Emp subjects common
        "BMSG_commonSubs", BMSG_commonSubs, 
        "BMSB_commonSubs", BMSB_commonSubs, 
        "LSubjectsAll", LSubjectsAll, "subjectIDsLAll", subjectIDsLAll,
        )
