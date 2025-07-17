################################################################################
# Code for plotting inference results for Posterior Predictive Checks
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

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExp2Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp2Data/Figures/"
Path_Load = "data/Experiment2/clean/"

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]

# ----------------------------------------------------------------------
# Load group level inference data
# ----------------------------------------------------------------------
BMSdata = load(Path_Save * "BMS.jld2")
BMSdataLHat = load(Path_Save * "BMSbasedLHat.jld2")
BMSdataNa  = load(Path_Save * "BMSbasedNaFit.jld2")

subjectIDsL  = [BMSdata["subjectIDsLG"],BMSdata["subjectIDsLB"]]
BMSAllGroups = [BMSdata["BMSAllG"],BMSdata["BMSAllB"]]
BMSEmpGroups = [BMSdata["BMSG"],   BMSdata["BMSB"]]

# ----------------------------------------------------------------------
# Rooms
# ----------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

GorB = [1,0]
ProcessedDataAll = []
for i_sub = subjectIDs
    @show i_sub
    Path_Load_ID = Path_Load_Inf * 
                        "inference_data_sub" * string(i_sub) * ".jld2"
    temp = load(Path_Load_ID)
    chnAll_df = temp["chnAll_df"]
    chnAll_dfG = temp["chnAll_dfG"]; chnemp_dfG = temp["chnemp_dfG"]
    chnAll_dfB = temp["chnAll_dfB"]; chnemp_dfB = temp["chnemp_dfB"]

    ProcessedData = []

    for i_GorB = 1:2
        @show i_GorB
        chNa_df = BMSdataNa["chnA_dfsAll"][i_GorB][
                                            subjectIDs .== i_sub][1]        
        # ----------------------------------------------------------------------
        # Inferring preferences
        # ----------------------------------------------------------------------
        df = dataDF[dataDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        df = df[df.Gtrials .== GorB[i_GorB],:]
        
        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]  
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1
        
        # ----------------------------------------------------------------------
        # model selection
        # ----------------------------------------------------------------------
        pmAll = BMSAllGroups[i_GorB].exp_M[subjectIDs .== i_sub,:][:]
        mAll_hat = findmax(pmAll)[2]

        if i_sub ∈ subjectIDsL[i_GorB]
            pm = BMSEmpGroups[i_GorB].exp_M[subjectIDsL[i_GorB] .== 
                                                            i_sub,:][:]
            m_hat = findmax(pm)[2]
            df_temp = BMSdataLHat["chnemp_dfsAll"][i_GorB][
                                            subjectIDsL[i_GorB] .== i_sub][1]
            βs = df_temp.β
            if m_hat == 1
                ls = df_temp.l0
            elseif m_hat == 2
                ls = ones(length(df_temp.l0))
            elseif m_hat == 3
                ls = df_temp.l2
            end
            l_hat = mean(ls)
            β_hat = mean(βs)

            emp_vals_hat  = zeros(length(ls),N_rooms)
            emp_rvals_hat = zeros(length(ls),N_rooms)
            for i = eachindex(ls)
                emp_model_hat = emplK(ls[i],1.,1)
                for i_room = 1:N_rooms
                    p, StateS, StateSDict, N_s = 
                        gold_env_setup(Prooms[i_room], ΔState, 
                                        ΔStateDict, Xmax, Ymax)
                    emp_vals_hat[i,i_room] = βs[i] * 
                                p2empCat(p, StateSDict[(0,0)], emp_model_hat)
                end
                emp_rvals_hat[i,:] = Rank(emp_vals_hat[i,:])
            end
        else
            pm = BMSEmpGroups[i_GorB].exp_M[1,:] .* NaN
            m_hat = NaN

            βs = chNa_df.βa
            emp_vals_hat  = zeros(length(βs),N_rooms)
            for i = eachindex(βs)
                for i_room = 1:N_rooms
                    emp_vals_hat[i,i_room] = βs[i] * size(Prooms[i_room])[1]
                end
            end

            ls = [chnemp_dfG,chnemp_dfB][i_GorB] .* NaN
            l_hat = NaN
            β_hat = NaN
        end
        

        # ----------------------------------------------------------------------
        # Room preferences
        # ----------------------------------------------------------------------
        prefMat_raw   = zeros(N_rooms, N_rooms) .* NaN
        prefMat_model = zeros(N_rooms, N_rooms) .* NaN
        for i1_paper = 1:N_rooms
            for i2_paper = (i1_paper+1):N_rooms
                i1 = PaperRoomOrder[i1_paper]
                i2 = PaperRoomOrder[i2_paper]
                t_temp = [findfirst(==([i1,i2]), Xinds),
                          findfirst(==([i2,i1]), Xinds)]
                t_temp = t_temp[isnothing.(t_temp) .== 0]
                if length(t_temp) > 0
                    mp1 = mean((df.chosenroom[t_temp].+1) .== i1)
                    mp2 = 1 - mp1
                    prefMat_raw[i1_paper, i2_paper] = mp1 - mp2
                    prefMat_raw[i2_paper, i1_paper] = mp2 - mp1

                    v1s = emp_vals_hat[:,i1]; 
                    v2s = emp_vals_hat[:,i2]; Δvs = v2s .- v1s

                    p1s = 1 ./ (1 .+ exp.(Δvs)); p2s = 1 .- p1s
                    mp1 = mean(p1s); dp1 = std(p1s)
                    mp2 = mean(p2s); dp2 = std(p2s)
                    prefMat_model[i1_paper, i2_paper] = mp1 - mp2
                    prefMat_model[i2_paper, i1_paper] = mp2 - mp1
                end
            end
        end
        # ----------------------------------------------------------------------
        # pushing
        # ----------------------------------------------------------------------
        push!(ProcessedData,
            (; pmAll = pmAll, mAll_hat = mAll_hat,
                pm = pm, m_hat = m_hat, l_hat = l_hat, β_hat = β_hat,
                prefMat_model = prefMat_model, prefMat_raw = prefMat_raw))
    end
    push!(ProcessedDataAll,ProcessedData)
end


# ----------------------------------------------------------------------
# Sepeate trial types
# ----------------------------------------------------------------------
GorB_labels = ["GTrials","BTrials"]
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    # ----------------------------------------------------------------------
    # choosing L subjects
    # ----------------------------------------------------------------------
    ProcessedData = [d[i_GorB] for d = ProcessedDataAll]
    BMSmAllHat = [BMSdata["mAll_hatG"],BMSdata["mAll_hatB"]][i_GorB]
    EmplProcessedData = ProcessedData[BMSmAllHat .== 3]
    BMSmHat = [findmax(BMSEmptemp.exp_M[i,:])[2] for 
                        i = 1:size(BMSEmptemp.exp_M)[1]] 

    # ----------------------------------------------------------------------
    # room preference plotting + action group
    # ----------------------------------------------------------------------
    Legends = ["L < 1","L = 1","L > 1"]

    y_raw   = [d.prefMat_raw for d = EmplProcessedData]
    y_model = [d.prefMat_model for d = EmplProcessedData]
    ms = BMSmHat
    
    for m = 1:3
        fig = figure(figsize=(12,6))
        if sum(ms .== m) > 0
            n = sum(ms .== m)
            my_raw   = mean(y_raw[ms .== m])
            my_model = mean(y_model[ms .== m])
            ρ = cor(my_raw[:][isnan.(my_raw[:]) .== 0],
                    my_model[:][isnan.(my_raw[:]) .== 0])

            Y = my_raw
            ax = subplot(1,2,1)
            cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
            fig.colorbar(cp, ax=ax)
            ax.set_xticks(0:(N_rooms - 1)); 
            ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_yticks(0:(N_rooms - 1)); 
            ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_title("Data; P(RY - RX); " * Legends[m] * 
                                                    "; n = " * string(n))

            Y = my_model
            ax = subplot(1,2,2)
            cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
            fig.colorbar(cp, ax=ax)
            ax.set_xticks(0:(N_rooms - 1)); 
            ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_yticks(0:(N_rooms - 1)); 
            ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
            ax.set_title("Model; P(RY - RX); r = " * string(round(ρ,digits=3)))
        end
        tight_layout()
        savefig(Path_Save * "PrefMat_M" * string(m) * "_" * 
                        GorB_labels[i_GorB] * ".pdf")
        savefig(Path_Save * "PrefMat_M" * string(m) * "_" * 
                            GorB_labels[i_GorB] * ".png")
        savefig(Path_Save * "PrefMat_M" * string(m) * "_" * 
                            GorB_labels[i_GorB] * ".svg")
    end
    
    my_raw   = mean([d.prefMat_raw for d = ProcessedData][BMSmAllHat .== 2]);
    my_model =  mean([d.prefMat_model for d = ProcessedData][BMSmAllHat .== 2]);
    n = sum(BMSmAllHat .== 2)

    fig = figure(figsize=(12,6))
    ρ = cor(my_raw[:][isnan.(my_raw[:]) .== 0],
            my_model[:][isnan.(my_raw[:]) .== 0])

    Y = my_raw
    ax = subplot(1,2,1)
    cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:(N_rooms - 1)); 
    ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
    ax.set_yticks(0:(N_rooms - 1)); 
    ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
    ax.set_title("Data; P(RY - RX); N-act; n = " * string(n))

    Y = my_model
    ax = subplot(1,2,2)
    cp = ax.imshow(Y, vmin = -1, vmax = 1, cmap="RdBu")
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:(N_rooms - 1)); 
    ax.set_xticklabels(["R" * string(i) for i = 1:N_rooms])
    ax.set_yticks(0:(N_rooms - 1)); 
    ax.set_yticklabels(["R" * string(i) for i = 1:N_rooms])
    ax.set_title("Model; P(RY - RX); r = " * string(round(ρ,digits=3)))

    tight_layout()
    savefig(Path_Save * "PrefMat_MNa_" * 
                    GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "PrefMat_MNa_" * 
                        GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "PrefMat_MNa_" * 
                        GorB_labels[i_GorB] * ".svg")
end
