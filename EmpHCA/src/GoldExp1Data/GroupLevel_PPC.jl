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

Path_Save = "src/GoldExp1Data/Figures/GroupLevel/"
Path_Load_Inf = "src/GoldExp1Data/Figures/"
Path_Load = "data/Experiment1/clean/"

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]

# --------------------------------------------------------------------------
# Load group level inference data
# --------------------------------------------------------------------------
BMSdata = load(Path_Save * "BMS.jld2")
BMSdataLHat = load(Path_Save * "BMSbasedLHat.jld2")

# --------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

ProcessedData = []
for i_sub = subjectIDs
    @show i_sub
    # --------------------------------------------------------------------------
    # Inference data
    # --------------------------------------------------------------------------
    temp = load(Path_Load_Inf * "inference_data_sub" * string(i_sub) * ".jld2")
    chn_df = temp["chn_df"]
    chnAll_df = temp["chnAll_df"]
    chnemp_df = temp["chnemp_df"]
        
    # --------------------------------------------------------------------------
    # Inferring preferences
    # --------------------------------------------------------------------------
    df = dataDF[dataDF.subject .== i_sub, :]
    df = df[df.timeout .== false, :]

    Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]
    
    as_inds = df.chosenroom .+ 1
    as = df.action .+ 1

    # --------------------------------------------------------------------------
    # model selection
    # --------------------------------------------------------------------------
    pmAll = BMSdata["BMSAll"].exp_M[subjectIDs .== i_sub,:][:]
    mAll_hat = findmax(pmAll)[2]
    
    if i_sub ∈ BMSdata["subjectIDs_L"]
        pm = BMSdata["BMS"].exp_M[BMSdata["subjectIDs_L"] .== i_sub,:][:]
        m_hat = findmax(pm)[2]

        df_temp = BMSdataLHat["chnemp_dfs"][
                            BMSdataLHat["subjectIDs_L"] .== i_sub][1]
        βs = df_temp.β
        if m_hat == 1
            ls = df_temp.l0
        elseif m_hat == 2
            ls = ones(lenght(df_temp.l0))
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
                    gold_env_setup(Prooms[i_room], ΔState, ΔStateDict, 
                                        Xmax, Ymax)
                emp_vals_hat[i,i_room] = βs[i] * p2empCat(p, StateSDict[(0,0)], emp_model_hat)
            end
            emp_rvals_hat[i,:] = Rank(emp_vals_hat[i,:])
        end
    else
        pm = BMSdata["BMS"].exp_M[1,:] .* NaN
        m_hat = NaN

        
        βs = chnAll_df.βa
        emp_vals_hat  = zeros(length(βs),N_rooms)
        for i = eachindex(βs)
            for i_room = 1:N_rooms
                emp_vals_hat[i,i_room] = βs[i] * size(Prooms[i_room])[1]
            end
        end

        ls = chnemp_df.l0 .* NaN
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

    # --------------------------------------------------------------------------
    # pushing
    # --------------------------------------------------------------------------
    push!(ProcessedData,
        (; pmAll = pmAll, mAll_hat = mAll_hat,
            pm = pm, m_hat = m_hat, l_hat = l_hat, β_hat = β_hat,
            prefMat_model = prefMat_model, prefMat_raw = prefMat_raw))
end


# --------------------------------------------------------------------------
# choosing L subjects
# --------------------------------------------------------------------------
BMSmAllHat = [findmax(BMSdata["BMSAll"].exp_M[i,:])[2] for 
                    i = 1:size(BMSdata["BMSAll"].exp_M)[1]]
EmplProcessedData = ProcessedData[BMSmAllHat .== 3] 
BMSmHat = [findmax(BMSdata["BMS"].exp_M[i,:])[2] for 
                    i = 1:size(BMSdata["BMS"].exp_M)[1]]

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
        ax.set_title("Data; P(RY - RX); " * Legends[m] * "; n = " * string(n))

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
    savefig(Path_Save * "PrefMat_M" * string(m) * ".pdf")
    savefig(Path_Save * "PrefMat_M" * string(m) * ".png")
    savefig(Path_Save * "PrefMat_M" * string(m) * ".svg")
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
savefig(Path_Save * "PrefMat_MNa.pdf")
savefig(Path_Save * "PrefMat_MNa.png")
savefig(Path_Save * "PrefMat_MNa.svg")
