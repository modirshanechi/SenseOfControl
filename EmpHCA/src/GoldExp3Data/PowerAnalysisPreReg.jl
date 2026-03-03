################################################################################
# Using bootstrapping on the data from Experiment 2 to run the power-analysis
# for Experiment 3; see the pre-registration at https://osf.io/rnf8v/
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

# loading data of Experiment 2
Path_Load_Inf = "src/GoldExp2Data/Figures/"
Path_Load = "data/Experiment2/clean/"

# Evaluation for Experiment 3
Path_Save = "src/GoldExp3Data/Figures/PowerAnalysis/"


# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
ExcDF  = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs  = ExcDF.subject[ExcDF.task_outliers .== 0]

# ------------------------------------------------------------------------
# Calculating model evidence for all Subjects
# ------------------------------------------------------------------------
ProcessedData = []
for i_sub = subjectIDs
    Path_Load_ID = Path_Load_Inf * "inference_data_sub" * 
                                        string(i_sub) * ".jld2"

    @show i_sub
    # ------------------------------------------------------------------------
    # Inference data
    # ------------------------------------------------------------------------
    temp = load(Path_Load_ID)
    chnAll_dfG = temp["chnAll_dfG"]
    df = dataDF[dataDF.subject .== i_sub,:]
    
    # ------------------------------------------------------------------------
    # model selection
    # ------------------------------------------------------------------------
    pmAllG = [sum(chnAll_dfG.m .== j) for j = 1:4]; 
    pmAllG = pmAllG ./ sum(pmAllG)

    push!(ProcessedData,
        (;  pmAllG = pmAllG, 
            i_sub = i_sub, Cond = df.GB_condition[1]))
end
L_matrix = Matrix(hcat([d.pmAllG for d = ProcessedData]...)')
L_matrix .= log.(max.(L_matrix,0.001)); 
N_sub, N_model = size(L_matrix)


# ------------------------------------------------------------------------
# bootstrapping subjects to estimate power
# ------------------------------------------------------------------------
L = 200
N_samp_set = 80:10:140
power_set = []; dpower_set = []
for N_samp = N_samp_set
    @show N_samp
    pxp_set = []
    for l = 1:L
        @show l
        idx = rand(1:N_sub, N_samp)
        L_matrix_temp = L_matrix[idx,:]

        BMSAllG = MCMC_BMS_Statistics(L_matrix_temp; 
                        N_Sampling = Int(1e5), 
                        N_Sampling_BOR = Int(1e5), N_Chains = 50, α = 1. ./ N_model,
                        test_plotting = false)
        push!(pxp_set, BMSAllG.pxp[3])
    end
    power = mean(pxp_set .> 0.95); dpower = std(pxp_set .> 0.95) / sqrt(L)
    
    println("N = " * string(N_samp))
    println("power = " * string(power) * " +- " * string(dpower))
    println("---------------")
    push!(power_set,power); push!(dpower_set,dpower)


    save(Path_Save * "Ns_" * string(N_samp) * ".jld2", 
            "pxp_set", pxp_set, "power", power, "dpower", dpower, "N_samp", N_samp)
end

# ------------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------------
figure(figsize=(5,5))
ax = subplot(1,1,1)
ax.plot(N_samp_set, power_set, color="k")
ax.fill_between(N_samp_set, power_set .- dpower_set, 
                            power_set .+ dpower_set, color = "k", alpha = 0.3)
ax.plot([N_samp_set[1],N_samp_set[end]], [1,1] .* 0.8, "--k")
ax.set_xlim([N_samp_set[1],N_samp_set[end]]); ax.set_ylim([0,1])
ax.set_ylabel("power")
ax.set_xlabel("number of samples")
tight_layout()
savefig(Path_Save * "power_analysis.pdf")
savefig(Path_Save * "power_analysis.png")
savefig(Path_Save * "power_analysis.svg")

