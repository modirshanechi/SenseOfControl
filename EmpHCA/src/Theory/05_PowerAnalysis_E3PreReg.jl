################################################################################
# Using bootstrapping on the data from Experiment 2 to run the power-analysis
# for Experiment 3; see the pre-registration at https://osf.io/rnf8v/
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random
using DataFrames
using CSV
using JLD2
using AdvancedMH
using Statistics

import StatsPlots

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)

SavePath = "src/Theory/Figures/PowerAnalysis/"

# ------------------------------------------------------------------------
# On whether the analysis should be done based on the results of Experiment 2
# or to confirm the power based on the results of Experiment 3
# ------------------------------------------------------------------------
ifposthoc = true

if !ifposthoc
    # ------------------------------------------------------------------------
    # Loading model evidence for gold trials of Experiment 2
    # ------------------------------------------------------------------------
    expspec = ExperimentSpecification(2)
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    L_matrix = BMSdata["L_matrixAll"][1]
    nametag = ""
else
    # ------------------------------------------------------------------------
    # Loading model evidence for Experiment 3
    # ------------------------------------------------------------------------
    expspec = ExperimentSpecification(3)
    BMSdata = load(expspec.fig_path * "BMS.jld2")
    L_matrix = BMSdata["L_matrixAll"]
    nametag = "_posthoc"
end

N_sub, N_model = size(L_matrix)

# ------------------------------------------------------------------------
# bootstrapping subjects to estimate power
# ------------------------------------------------------------------------
model_of_interest = 3 # considering Emp-l as the target model of inferece for pxp
L = 200 # number of bootstrapping samples
# L = 10 # number of bootstrapping samples

if !ifposthoc
    N_samp_set = [170, 180, 190]
else
    N_samp_set = [N_sub]
end

power_set95 = Vector{Float64}([]); dpower_set95 = Vector{Float64}([])
power_set90 = Vector{Float64}([]); dpower_set90 = Vector{Float64}([])
power_set80 = Vector{Float64}([]); dpower_set80 = Vector{Float64}([])
for N_samp = N_samp_set
    println("---------------")
    println("---------------")
    @show N_samp
    pxp_set = Vector{Float64}([])
    for l = 1:L
        @show l
        idx = rand(rng, 1:N_sub, N_samp)
        L_matrix_temp = L_matrix[idx,:]

        BMSAllG = MCMC_BMS_Statistics(rng, L_matrix_temp; 
                        N_Sampling = Int(5e5), 
                        N_Sampling_BOR = Int(1e5), N_Chains = 10, α = 1. ./ N_model,
                        N_burn_in = 5000, N_thin = 1,
                        test_plotting = false, run_diagnostics = false)
        push!(pxp_set, BMSAllG.pxp[model_of_interest])
        @show BMSAllG.pxp
    end
    power95 = mean(pxp_set .> 0.95); dpower95 = std(pxp_set .> 0.95) / sqrt(L)
    power90 = mean(pxp_set .> 0.90); dpower90 = std(pxp_set .> 0.90) / sqrt(L)
    power80 = mean(pxp_set .> 0.80); dpower80 = std(pxp_set .> 0.80) / sqrt(L)

    println("---------------")
    println("N = " * string(N_samp))
    println("power based on 95% pxp = " * string(power95) * " +- " * string(dpower95))
    println("power based on 90% pxp = " * string(power90) * " +- " * string(dpower90))
    println("power based on 80% pxp = " * string(power80) * " +- " * string(dpower80))
    println("---------------")
    push!(power_set95,power95); push!(dpower_set95,dpower95)
    push!(power_set90,power90); push!(dpower_set90,dpower90)
    push!(power_set80,power80); push!(dpower_set80,dpower80)


    save(SavePath * "Ns_" * string(N_samp) * nametag * ".jld2", 
            "pxp_set", pxp_set, "N_samp", N_samp,
            "power95", power95, "dpower95", dpower95, 
            "power90", power90, "dpower90", dpower90, 
            "power80", power80, "dpower80", dpower80)
end

if !ifposthoc
    # ------------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------------
    Yset = [power_set95, power_set90, power_set80]
    dYset = [dpower_set95, dpower_set90, dpower_set80]
    YName = ["95pxp", "90pxp", "80pxp"]
    for i_y = eachindex(Yset)
        y = Yset[i_y]; dy = dYset[i_y]; yname = YName[i_y]
        figure(figsize=(5,5))
        ax = subplot(1,1,1)
        ax.plot(N_samp_set, y, color="k")
        ax.fill_between(N_samp_set, y .- dy, y .+ y, color = "k", alpha = 0.3)
        ax.plot([N_samp_set[1],N_samp_set[end]], [1,1] .* 0.8, "--k")
        ax.set_xlim([N_samp_set[1],N_samp_set[end]]); ax.set_ylim([0,1])
        ax.set_ylabel("power for " * yname)
        ax.set_xlabel("number of samples")
        tight_layout()
        savefig(SavePath * "power_analysis_" * yname * ".pdf")
        savefig(SavePath * "power_analysis_" * yname * ".png")
        savefig(SavePath * "power_analysis_" * yname * ".svg")
    end
end