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

Path_Save = "src/GoldExp1Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp1Data/Figures/"
Path_Load = "data/Experiment1/clean/"

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

ProcessedData = []
for i_sub = subjectIDs
    @show i_sub
    # ------------------------------------------------------------------------------
    # Inference data
    # ------------------------------------------------------------------------------
    temp = load(Path_Load_Inf * "inference_data_sub" * string(i_sub) * ".jld2")
    chn_df = temp["chn_df"]
    chnAll_df = temp["chnAll_df"]
    chnemp_df = temp["chnemp_df"]
    @show size(chn_df)
    @show size(chnAll_df)
    @show size(chnemp_df)
    # ------------------------------------------------------------------------------
    # Inferring preferences
    # ------------------------------------------------------------------------------
    df = dataDF[dataDF.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    # ------------------------------------------------------------------------------
    # model selection: random, Nact, Empl, θ
    # ------------------------------------------------------------------------------
    pmAll = [sum(chnAll_df.m .== j) for j = 1:4]; pmAll = pmAll ./ sum(pmAll)
    mAll_hat = findmax(pmAll)[2]

    # ------------------------------------------------------------------------------
    # model selection: l0, l1, l2
    # ------------------------------------------------------------------------------
    pm = [sum(chnemp_df.m .== j) for j = 1:3]; pm = pm ./ sum(pm)
    m_hat = findmax(pm)[2]
    
    push!(ProcessedData,
        (; pmAll = pmAll, mAll_hat = mAll_hat,
            pm = pm, m_hat = m_hat))
end


# ------------------------------------------------------------------------------
# Random effect general model selection
# ------------------------------------------------------------------------------
L_matrix = Matrix(hcat([d.pmAll for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); N_model = size(L_matrix)[2]
BMSAll = MCMC_BMS_Statistics(L_matrix; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 40, α = 1. ./ N_model)

# ------------------------------------------------------------------------------
# Random effect l-based model selection
# ------------------------------------------------------------------------------
mAll_hat = [findmax(BMSAll.exp_M[i,:])[2] for i = 1:size(BMSAll.exp_M)[1]]
LSubjects = (1:length(mAll_hat))[mAll_hat .== 3]
subjectIDs_L = subjectIDs[LSubjects]

L_matrix = Matrix(hcat([d.pm for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); N_model = size(L_matrix)[2]
L_matrix = L_matrix[LSubjects,:]

BMS = MCMC_BMS_Statistics(L_matrix; N_Sampling = Int(1e5), 
                N_Sampling_BOR = Int(1e5), N_Chains = 40, α = 1. ./ N_model)

save(Path_Save * "BMS.jld2", "BMSAll",BMSAll, "BMS", BMS,
        "LSubjects", LSubjects, "subjectIDs", subjectIDs, 
        "subjectIDs_L", subjectIDs_L)
