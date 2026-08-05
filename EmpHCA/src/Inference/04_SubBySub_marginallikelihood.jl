################################################################################
# Code for hierarchical inference for models
#   --> General model selection (Random vs Na vs Emp-l vs BradleyTerry) is done
#       using BIC approximation of log evidence + MCMC custom code for BMS
#   --> Emp-l model selection (l<1 vs l=1 vs l>1) is done using the MCMC-based
#       estimate of log-evidence + MCMC custom code for BMS
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


# ----------------------------------------------------------------------
# inference information
# ----------------------------------------------------------------------
N_Chains = 10
N_Sampling = Int(5e6)
N_Sampling_BOR = Int(5e5)
N_burn_in = 5000
N_thin = 100
test_plotting = false

# ----------------------------------------------------------------------
# rooms
# ----------------------------------------------------------------------
Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax = room_information()

N_modelAll = length(modelspecs_BIC)

# ----------------------------------------------------------------------
# range-of-l model selection (l<1 vs l=1 vs l>1), restricted to subjects
# classified as Emp-l (model 3) or BradleyTerry (model 4) in the general
# model selection
# ----------------------------------------------------------------------
function LRangeBMS(mvalsEmp_list, BMSAll, rng; N_modelEmp = 3)
    # --------------------------------------------------------------
    # log-evidence estimation
    # --------------------------------------------------------------
    N_sub = length(mvalsEmp_list)
    P_matrixEmp = fill(NaN, N_sub, N_modelEmp)
    N_samplesEmp = length(mvalsEmp_list[1])
    for i = 1:N_sub
        mvals = mvalsEmp_list[i]
        P_matrixEmp[i,:] = [sum(mvals .== j) for j = 1:N_modelEmp];
        P_matrixEmp[i,:] = P_matrixEmp[i,:] ./ sum(P_matrixEmp[i,:])
    end
    L_matrixEmp= log.(max.(P_matrixEmp,1/N_samplesEmp))

    # --------------------------------------------------------------
    # Random effect l-based model selection
    # --------------------------------------------------------------
    mAll_hat = [findmax(BMSAll.exp_M[i,:])[2] for i = 1:size(BMSAll.exp_M)[1]]

    
    # on both general and Emp-l participants together
    EmpSubjects = (1:length(mAll_hat))[(mAll_hat .== 3) .|| (mAll_hat .== 4)]
    L_matrixEmp_temp = L_matrixEmp[EmpSubjects, :]
    BMSEmp = MCMC_BMS_Statistics(rng, L_matrixEmp_temp; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelEmp, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)
    
    # on only Emp-l participants
    EmpSubjectsalone = (1:length(mAll_hat))[(mAll_hat .== 3)]
    L_matrixEmp_temp = L_matrixEmp[EmpSubjectsalone, :]
    BMSEmpalone = MCMC_BMS_Statistics(rng, L_matrixEmp_temp; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelEmp, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)

    # on only general participants
    EmpSubjectsGen   = (1:length(mAll_hat))[(mAll_hat .== 4)]
    L_matrixEmp_temp = L_matrixEmp[EmpSubjectsGen, :]
    BMSEmpGen = MCMC_BMS_Statistics(rng, L_matrixEmp_temp; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelEmp, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)

    return (; BMSEmp = BMSEmp, EmpSubjects = EmpSubjects,
                BMSEmpalone = BMSEmpalone, EmpSubjectsalone = EmpSubjectsalone,
                BMSEmpGen = BMSEmpGen, EmpSubjectsGen = EmpSubjectsGen)
end


# ----------------------------------------------------------------------
# loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    rng = Xoshiro(2024)

    println("------------------------")
    @show i_exp
    println("------------------------")
    # ----------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec)
    ExcDF = data.ExcDF; selDF = data.selDF; subjectIDs = data.subjectIDs

    # ----------------------------------------------------------------------
    # load inference data
    # ----------------------------------------------------------------------
    infdata = [load(expspec.fig_path * "MCMC_sub" *
                    string(i_sub) * ".jld2") for i_sub = subjectIDs]

    if !expspec.has_gb_split
        # fitting MLE and evaluating BIC for the general model selection
        L_matrixAll = bic_logp_matrix(selDF, subjectIDs, infdata,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax, modelspecs_BIC, i_exp)
        # using MCMC-based model selection for emp l group
        mvalsEmp_list = [d["MSresultsEmp"].chnemp_df.m for d = infdata]

        # emptying memory allocated to infdata
        infdata = nothing; GC.gc()

        # running hierarchical inference for general model selection
        BMSAll = MCMC_BMS_Statistics(rng, L_matrixAll; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelAll, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)
        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS all:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSAll.r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSAll.M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSAll.M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSAll.d_BOR / BMSAll.BOR)

        # running hierarchical inference for Emp-l model selection
        res = LRangeBMS(mvalsEmp_list, BMSAll, rng)
        
        BMSEmp = res.BMSEmp; EmpSubjects = res.EmpSubjects
        BMSEmpalone = res.BMSEmpalone; BMSEmpGen = res.BMSEmpGen
        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS Emp-l:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSEmp.r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSEmp.M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSEmp.M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSEmp.d_BOR / BMSEmp.BOR)

        EmpsubjectIDs = subjectIDs[EmpSubjects]
    else
        # fitting MLE and evaluating BIC for the general model selection
        L_matrixAllG = bic_logp_matrix(selDF, subjectIDs, infdata,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax, modelspecs_BIC, i_exp;
                        gb_idx = 1)
        L_matrixAllB = bic_logp_matrix(selDF, subjectIDs, infdata,
                        Prooms, ΔState, ΔStateDict, N_rooms, Ymax, Xmax, modelspecs_BIC, i_exp;
                        gb_idx = 2)
        # using MCMC-based model selection for emp l group
        mvalsEmp_listG = [d["MSresultsEmp"][1].chnemp_df.m for d = infdata]        
        mvalsEmp_listB = [d["MSresultsEmp"][2].chnemp_df.m for d = infdata]

        # emptying memory allocated to infdata
        infdata = nothing; GC.gc()

        # running hierarchical inference for general model selection
        BMSAllG = MCMC_BMS_Statistics(rng, L_matrixAllG; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelAll, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)
        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS all for gold trials:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSAllG.r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSAllG.M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSAllG.M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSAllG.d_BOR / BMSAllG.BOR)

        BMSAllB = MCMC_BMS_Statistics(rng, L_matrixAllB; N_Sampling = N_Sampling,
                    N_Sampling_BOR = N_Sampling_BOR, N_Chains = N_Chains, 
                    α = 1. ./ N_modelAll, N_burn_in = N_burn_in, N_thin = N_thin,
                    test_plotting = test_plotting)

        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS all for bomb trials:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSAllB.r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSAllB.M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSAllB.M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSAllB.d_BOR / BMSAllB.BOR)
        
        BMSAll = [BMSAllG, BMSAllB]
        L_matrixAll = [L_matrixAllG, L_matrixAllB]

        # running hierarchical inference for Emp-l model selection
        resG = LRangeBMS(mvalsEmp_listG, BMSAll[1], rng)
        resB = LRangeBMS(mvalsEmp_listB, BMSAll[2], rng)

        BMSEmp = [resG.BMSEmp, resB.BMSEmp]
        BMSEmpalone = [resG.BMSEmpalone, resB.BMSEmpalone]
        BMSEmpGen = [resG.BMSEmpGen, resB.BMSEmpGen]

        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS Emp-l for gold trials:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSEmp[1].r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSEmp[1].M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSEmp[1].M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSEmp[1].d_BOR / BMSEmp[1].BOR)

        println("------------------------------------")
        println("------------------------------------")
        @show i_exp
        println("MCMC diagnosis for BMS Emp-l for bomb trials:")
        println("------------------------------------")
        println("------------------------------------")
        @show BMSEmp[2].r_diagnostics
        println("Fraction of converged M-indicators:")
        @show mean(BMSEmp[2].M_diagnostics.pass)
        println("Fraction of constant M-indicators:")
        @show mean(BMSEmp[2].M_diagnostics.constant)
        println("BOR relative error:")
        @show mean(BMSEmp[2].d_BOR / BMSEmp[2].BOR)

        EmpSubjects = [resG.EmpSubjects, resB.EmpSubjects]
        EmpsubjectIDs = [subjectIDs[resG.EmpSubjects], subjectIDs[resB.EmpSubjects]]
    end

    save(expspec.fig_path * "BMS.jld2",
            "BMSAll",BMSAll, "BMSEmp", BMSEmp, "L_matrixAll", L_matrixAll,
            "BMSEmpalone", BMSEmpalone, "BMSEmpGen", BMSEmpGen, 
            "EmpSubjects", EmpSubjects, "EmpsubjectIDs", EmpsubjectIDs)
end

