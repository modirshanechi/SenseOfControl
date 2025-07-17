################################################################################
# Code for plotting inference results seprately for the two blocks
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

room_pairs = [[2,6],[2,3],[10,9],[5,8],[5,7],[6,7]]

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
        # rt
        # ----------------------------------------------------------------------
        rt = df.rt ./ 1000
        valid_inds = rt .< 10
        rt = rt[valid_inds]
        dv  = abs.([size(Prooms[x[1]])[1] - size(Prooms[x[2]])[1] 
                            for x = Xinds])[valid_inds]
        ρrt_contNa = cor(dv,rt)
        rank_ρrt_contNa = cor(Rank(dv),Rank(rt))

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

            rt = df.rt ./ 1000
            valid_inds = rt .< 10
            rt = rt[valid_inds]
            m_emp_vals_hat = mean(emp_vals_hat,dims=1)[:]
            dv  = abs.([m_emp_vals_hat[x[1]] - m_emp_vals_hat[x[2]] 
                            for x = Xinds])[valid_inds]
            ρrt = cor(dv,rt)
            rank_ρrt = cor(Rank(dv),Rank(rt))
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

            ρrt      = ρrt_contNa
            rank_ρrt = rank_ρrt_contNa
        end
        

        # ----------------------------------------------------------------------
        # Room 2 versus 3 and room 9 versus 10
        # ----------------------------------------------------------------------
        room_pref = []
        for i = eachindex(room_pairs)
            i1, i2 = room_pairs[i]

            t_temp = [findfirst(==([i1,i2]), Xinds),
                      findfirst(==([i2,i1]), Xinds)]
            t_temp = t_temp[isnothing.(t_temp) .== 0]

            Legends = ["Room " * string(i1), "Room " * string(i2)]

            if length(t_temp) > 0
                mp1 = mean((df.chosenroom[t_temp].+1) .== i1)
                mp2 = 1 - mp1
                mp_raw = [mp1, mp2]

                v1s = emp_vals_hat[:,i1]; 
                v2s = emp_vals_hat[:,i2]; Δvs = v2s .- v1s

                p1s = 1 ./ (1 .+ exp.(Δvs)); p2s = 1 .- p1s
                mp1 = mean(p1s); dp1 = std(p1s)
                mp2 = mean(p2s); dp2 = std(p2s)
                mp_model = [mp1, mp2]
            else
                mp_raw   = [NaN, NaN]
                mp_model = [NaN, NaN]
            end

            push!(room_pref, (; pair = room_pairs[i], mp_raw = mp_raw,
                                mp_model = mp_model))
        end
        # ----------------------------------------------------------------------
        # pushin
        # ----------------------------------------------------------------------
        push!(ProcessedData,
            (; pmAll = pmAll, mAll_hat = mAll_hat,
                pm = pm, m_hat = m_hat, l_hat = l_hat, β_hat = β_hat,
                ρrt = ρrt, rank_ρrt = rank_ρrt, 
                ρrt_contNa = ρrt_contNa, rank_ρrt_contNa = rank_ρrt_contNa,
                room_pref = room_pref))
    end
    push!(ProcessedDataAll,ProcessedData)
end

# ----------------------------------------------------------------------
# model selection plotting (all subjects)
# ----------------------------------------------------------------------
Y = BMSdata["BMSAll"].exp_M; 
Y_names = ["S" * string(s) for s = subjectIDs]
x_names = ModelNames.MAllE2
y = mean(Y,dims=1)[:]; i_winner = findmax(y)[2]
# Heatmap
sort_inds = sortperm(Y[:,i_winner])
y = Y[sort_inds,:]; y_names = Y_names[sort_inds]
fig = figure(figsize = (3,12)); ax = subplot(1,1,1)
cp = ax.imshow(y, cmap="binary_r",vmin=0.,vmax=1.,aspect="auto")
ax.set_xticks(0:(size(y)[2]-1)); 
ax.set_xticklabels(x_names,fontsize=9,rotation = 90)
ax.set_yticks(0:(size(y)[1]-1)); 
ax.set_yticklabels(y_names,fontsize=9)
ax.set_title("all subjects")
fig.colorbar(cp, ax=ax)
tight_layout()
savefig(Path_Save * "ModelSelAll_expM.pdf")
savefig(Path_Save * "ModelSelAll_expM.png")
savefig(Path_Save * "ModelSelAll_expM.svg")

# average 
y = BMSdata["BMSAll"].exp_r; 
dy = BMSdata["BMSAll"].d_exp_r; x = 1:length(y)
fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
ax.bar(x,y, color="k",alpha=0.7)
ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
            linestyle="dashed",linewidth=1,color="k")
title("Posterior Probabilities for Different Models")
ax.set_xticks(x)
ax.set_xticklabels(x_names,fontsize=9, rotation=-45)
ax.set_ylabel("E[P(model) | Data ]")
ax.set_xlim([x[1]-1,x[end]+1])
ax.set_ylim([0,1.0])

y = BMSdata["BMSAll"].pxp; x = 1:length(y)
ax = subplot(1,2,2)
ax.bar(x,y, color="k")
ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
            linestyle="dashed",linewidth=1,color="k")
title("Protected exceedence probabilities")
ax.set_xticks(x)
ax.set_xticklabels(x_names,fontsize=9, rotation=-45)
ax.set_ylabel("P[r_m > r_m' | Data ]")
ax.set_xlim([x[1]-1,x[end]+1])
ax.set_ylim([0,1.0])

tight_layout()
savefig(Path_Save * "ModelSelAll_expR.pdf")
savefig(Path_Save * "ModelSelAll_expR.png")
savefig(Path_Save * "ModelSelAll_expR.svg")


# ----------------------------------------------------------------------
# Sepeate trial types
# ----------------------------------------------------------------------
GorB_labels = ["GTrials","BTrials"]
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    Y = BMSAlltemp.exp_M; 
    Y_names = ["S" * string(s) for s = subjectIDs]
    x_names = ModelNames.MAllE1
    y = mean(Y,dims=1)[:]; i_winner = findmax(y)[2]
    # Heatmap
    sort_inds = sortperm(Y[:,i_winner])
    y = Y[sort_inds,:]; y_names = Y_names[sort_inds]
    fig = figure(figsize = (3,12)); ax = subplot(1,1,1)
    cp = ax.imshow(y, cmap="binary_r",vmin=0.,vmax=1.,aspect="auto")
    ax.set_xticks(0:(size(y)[2]-1)); 
    ax.set_xticklabels(x_names,fontsize=9,rotation = 90)
    ax.set_yticks(0:(size(y)[1]-1)); 
    ax.set_yticklabels(y_names,fontsize=9)
    ax.set_title("all subjects; " * GorB_labels[i_GorB])
    fig.colorbar(cp, ax=ax)
    tight_layout()
    savefig(Path_Save * "ModelSelAll_expM_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "ModelSelAll_expM_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "ModelSelAll_expM_" * GorB_labels[i_GorB] * ".svg")

    # average 
    y = BMSAlltemp.exp_r; 
    dy = BMSAlltemp.d_exp_r; x = 1:length(y)
    fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
    ax.bar(x,y, color="k",alpha=0.7)
    ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Posterior Probabilities for Different Models; " * GorB_labels[i_GorB])
    ax.set_xticks(x)
    ax.set_xticklabels(x_names,fontsize=9)
    ax.set_ylabel("E[P(model) | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    y = BMSAlltemp.pxp; x = 1:length(y)
    ax = subplot(1,2,2)
    ax.bar(x,y, color="k")
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Protected exceedence probabilities; " * GorB_labels[i_GorB])
    ax.set_xticks(x)
    ax.set_xticklabels(x_names,fontsize=9)
    ax.set_ylabel("P[r_m > r_m' | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    tight_layout()
    savefig(Path_Save * "ModelSelAll_expR_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "ModelSelAll_expR_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "ModelSelAll_expR_" * GorB_labels[i_GorB] * ".svg")


    x_names = ModelNames.MEmp
    figure(figsize=(5,5))
    ax = subplot(1,1,1)
    Y = [BMSEmptemp.exp_M[i,:] for 
                            i = eachindex(BMSdata["subjectIDsLG"])]
    y_col = [findmax(y)[2] for y = Y]
    ax.plot([0,1],[0,0],"k",alpha = 0.2)
    ax.plot([0,0],[0,1],"k",alpha = 0.2)
    ax.plot([0,1],[1,0],"k",alpha = 0.2)
    ax.set_xlim([-0.05,1.05]); ax.set_ylim([-0.05,1.05]); 
    ax.set_aspect("equal", "box")
    for i = eachindex(Y)
        y = Y[i] .+ 0.03 .* (rand() - 0.5)
        ax.plot(y[1],y[3],".",color= MainColors.lcol[y_col[i]],alpha=0.5)
    end
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_ylabel("P(l > 1| data); " * GorB_labels[i_GorB])
    ax.set_xlabel("P(l < 1| data); " * GorB_labels[i_GorB])
    tight_layout()
    savefig(Path_Save * "ModelSel_expM_tri_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "ModelSel_expM_tri_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "ModelSel_expM_tri_" * GorB_labels[i_GorB] * ".svg")


    # average 
    y = BMSEmptemp.exp_r; 
    dy = BMSEmptemp.d_exp_r; x = 1:length(y)
    fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
    for i = eachindex(x)
        ax.bar(x[i],y[i], color=MainColors.lcol[i])
    end
    ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Posterior Probabilities for Different Models; " * GorB_labels[i_GorB])
    ax.set_xticks(x)
    ax.set_xticklabels(x_names,fontsize=9)
    ax.set_ylabel("E[P(model) | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    y = BMSEmptemp.pxp; x = 1:length(y)
    ax = subplot(1,2,2)
    for i = eachindex(x)
        ax.bar(x[i],y[i], color=MainColors.lcol[i])
    end
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Protected exceedence probabilities; " * GorB_labels[i_GorB])
    ax.set_xticks(x)
    ax.set_xticklabels(x_names,fontsize=9)
    ax.set_ylabel("P[r_m > r_m' | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    tight_layout()
    savefig(Path_Save * "ModelSel_expR_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "ModelSel_expR_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "ModelSel_expR_" * GorB_labels[i_GorB] * ".svg")

    # ----------------------------------------------------------------------
    # choosing L subjects
    # ----------------------------------------------------------------------
    ProcessedData = [d[i_GorB] for d = ProcessedDataAll]
    BMSmAllHat = [BMSdata["mAll_hatG"],BMSdata["mAll_hatB"]][i_GorB]
    EmplProcessedData = ProcessedData[BMSmAllHat .== 3]
    BMSmHat = [findmax(BMSEmptemp.exp_M[i,:])[2] for 
                        i = 1:size(BMSEmptemp.exp_M)[1]]

    # ----------------------------------------------------------------------
    # l-hat
    # ----------------------------------------------------------------------
    ps = 0.0:0.001:1.
    y = [d.l_hat for d = EmplProcessedData]
    x = [d.m_hat for d = EmplProcessedData]
    c = [d.pm[3] - d.pm[1] for d = EmplProcessedData]
    cl = PyPlot.cm.RdBu
    figure(figsize = (5,5))
    ax = subplot(1,1,1)
    for i = unique(x)
        ax.plot(ps, ps .^ mean(y[x .== i]), color = MainColors.lcol[i],lw=2.5)
    end
    for i = eachindex(x)
        ax.plot(ps, ps .^ y[i], color = MainColors.lcol[x[i]],alpha=0.1)
    end
    ax.plot(ps, ps, "--k", alpha = 0.5)
    ax.set_xlim([0,1]);ax.set_ylim([0,1])
    ax.set_aspect(1)
    ax.set_xlabel("p_max"); ax.set_ylabel("p_max^l-hat")
    tight_layout()
    savefig(Path_Save * "LHat_ps_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "LHat_ps_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "LHat_ps_" * GorB_labels[i_GorB] * ".svg")

    figure()
    ax = subplot(1,1,1)
    ax.plot([0,0],[-1.05,1.05],"--k")
    temp_min = findmin(log.(y))[1] - 0.05
    temp_max = findmax(log.(y))[1] + 0.05
    ax.plot([temp_min,temp_max],[0,0],"--k")
    ax.plot([temp_min,temp_max],[-1,-1],"--k", alpha = 0.1)
    ax.plot([temp_min,temp_max],[+1,+1],"--k", alpha = 0.1)
    for i = unique(x)
        ax.plot(log.(y[x .== i]),c[x .== i],".",color = MainColors.lcol[i], alpha=0.5)
    end
    ax.set_ylim([-1.05,1.05]); ax.set_xlim([temp_min,temp_max])
    ax.set_xlabel("log l-hat"); ax.set_ylabel("p(l>1) - p(l<1)")
    tight_layout()
    savefig(Path_Save * "LHat_pm_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "LHat_pm_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "LHat_pm_" * GorB_labels[i_GorB] * ".svg")

    # ----------------------------------------------------------------------
    # β-hat
    # ----------------------------------------------------------------------
    y = [d.β_hat for d = EmplProcessedData]
    y[y .> 15] .= 15
    x = [d.m_hat for d = EmplProcessedData]
    c = [d.pm[3] - d.pm[1] for d = EmplProcessedData]
    cl = PyPlot.cm.RdBu
    y_range = 0:0.1:15
    figure(figsize = (5,5))
    ax = subplot(1,1,1)
    for i = unique(x)
        ax.hist(y[x .== i], 
                y_range, 
                color = MainColors.lcol[i], alpha = 0.5)
    end
    ax.set_xlim([0,y_range[end]])
    ax.set_xlabel("β-hat"); ax.set_ylabel("count")
    tight_layout()
    savefig(Path_Save * "BHat_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "BHat_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "BHat_" * GorB_labels[i_GorB] * ".svg")


    # ----------------------------------------------------------------------
    # room preference plotting + action group
    # ----------------------------------------------------------------------
    Legends = ["L < 1","L = 1","L > 1"]
    for i = eachindex(room_pairs)
        i1, i2 = room_pairs[i]; x = 1:2;
        RoomLegends = ["Room " * string(i1), "Room " * string(i2)]
        
        figure(figsize=(12,6))  
        
        y = [d.room_pref[i].mp_raw for d = EmplProcessedData]
        ms = BMSmHat
        my = [zeros(2) .* NaN for m = 1:3]
        dy = [zeros(2) .* NaN for m = 1:3]
        for m = 1:3
            if sum(ms .== m) > 0
                my[m] = mean(y[ms .== m])
                dy[m] = std(y[ms .== m]) / sqrt(sum(ms .== m))
            end
            ax = subplot(2,4,m)
            ax.bar(x, my[m], color = MainColors.lcol[m])
            ax.errorbar(x,my[m],yerr=dy[m],color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
            ax.set_title(Legends[m])
            ax.set_ylabel("ratio of choice out of 2")
            ax.set_xticks(x); ax.set_xticklabels(RoomLegends)
            ax.set_ylim([0,1.]); ax.set_xlim([0,x[end]+1])
            if sum(ms .== m) > 0
                Y_test = hcat(y[ms .== m]...); Y_test = Y_test[1,:] .- Y_test[2,:]
                Test_result = OneSampleTTest(Y_test)
                @show Test_result
                pval = pvalue(Test_result)
                logBF = [BIC_OneSampleTTest(Y_test), -BIC_OneSampleTTest(Y_test)]
                ax.set_title(Legends[m] *
                        "; p:" * Func_pval_string(pval) *
                        ", lBF:" * Func_logBF_string(logBF[1]))
            end
        end

        y = [d.room_pref[i].mp_raw for d = ProcessedData];
        y = y[BMSmAllHat .== 2]
        my = mean(y); dy = std(y) / sqrt(length(y))
        ax = subplot(2,4,4)
        ax.bar(x, my, color = "k", alpha = 0.5)
        ax.errorbar(x,my,yerr=dy,color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        ax.set_ylabel("ratio of choice out of 2")
        ax.set_xticks(x); ax.set_xticklabels(RoomLegends)
        ax.set_ylim([0,1.]); ax.set_xlim([0,x[end]+1])
        Y_test = hcat(y...); Y_test = Y_test[1,:] .- Y_test[2,:]
        Test_result = OneSampleTTest(Y_test)
        @show Test_result
        pval = pvalue(Test_result)
        logBF = [BIC_OneSampleTTest(Y_test), -BIC_OneSampleTTest(Y_test)]
        ax.set_title("N-act" *
                "; p:" * Func_pval_string(pval) *
                ", lBF:" * Func_logBF_string(logBF[1]))

        # y = [d.room_pref[i].mp_inferred for d = EmplProcessedData]
        y = [d.room_pref[i].mp_model for d = EmplProcessedData]
        ms = BMSmHat
        my = [zeros(2) .* NaN for m = 1:3]
        dy = [zeros(2) .* NaN for m = 1:3]
        for m = 1:3
            if sum(ms .== m) > 0
                Y_test = hcat(y[ms .== m]...)
                my[m] = mean(y[ms .== m])
                dy[m] = std(y[ms .== m]) / sqrt(sum(ms .== m))
            end
            ax = subplot(2,4,m + 4)
            ax.bar(x, my[m], color = MainColors.lcol[m])
            ax.errorbar(x,my[m],yerr=dy[m],color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
            ax.set_title(Legends[m])
            # ax.set_ylabel("inferred preference")
            ax.set_ylabel("model preference")
            ax.set_xticks(x); ax.set_xticklabels(RoomLegends)
            ax.set_ylim([0,1.]); ax.set_xlim([0,x[end]+1])
            if sum(ms .== m) > 0
                Y_test = hcat(y[ms .== m]...); Y_test = Y_test[1,:] .- Y_test[2,:]
                Test_result = OneSampleTTest(Y_test)
                @show Test_result
                pval = pvalue(Test_result)
                logBF = [BIC_OneSampleTTest(Y_test), -BIC_OneSampleTTest(Y_test)]
                ax.set_title(Legends[m] *
                        "; p:" * Func_pval_string(pval) *
                        ", lBF:" * Func_logBF_string(logBF[1]))
            end
        end

        # y = [d.room_pref[i].mp_inferred for d = ProcessedData];
        y = [d.room_pref[i].mp_model for d = ProcessedData];
        y = y[BMSmAllHat .== 2]
        my = mean(y); dy = std(y) / sqrt(length(y))
        ax = subplot(2,4,4+4)
        ax.bar(x, my, color = "k", alpha = 0.5)
        ax.errorbar(x,my,yerr=dy,color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
        # ax.set_ylabel("inferred preference")
        ax.set_ylabel("model preference")
        ax.set_xticks(x); ax.set_xticklabels(RoomLegends)
        ax.set_ylim([0,1.]); ax.set_xlim([0,x[end]+1])
        Y_test = hcat(y...); Y_test = Y_test[1,:] .- Y_test[2,:]
        Test_result = OneSampleTTest(Y_test)
        @show Test_result
        pval = pvalue(Test_result)
        logBF = [BIC_OneSampleTTest(Y_test), -BIC_OneSampleTTest(Y_test)]
        ax.set_title("N-act" *
                "; p:" * Func_pval_string(pval) *
                ", lBF:" * Func_logBF_string(logBF[1]))
        tight_layout()

        savefig(Path_Save * "NactRoomPair_" * string(i1) * "_" * string(i2) * "_" * GorB_labels[i_GorB] * ".pdf")
        savefig(Path_Save * "NactRoomPair_" * string(i1) * "_" * string(i2) * "_" * GorB_labels[i_GorB] * ".png")
        savefig(Path_Save * "NactRoomPair_" * string(i1) * "_" * string(i2) * "_" * GorB_labels[i_GorB] * ".svg")
    end

    # ----------------------------------------------------------------------
    # l-hat
    # ----------------------------------------------------------------------
    figure(figsize=(14,5))
    y = [d.l_hat for d = EmplProcessedData]
    x = [d.m_hat for d = EmplProcessedData]
    x_rang = [0:0.05:1.,0.9:0.1:1.1,1.0:0.1:5]
    Legends = ["l<1","l=1","l>1"]
    for i = 1:3
        ax = subplot(1,3,i)
        if sum(x .== i) > 0
            my = round(mean(y[x .== i]),digits=3)
            dy = round(std(y[x .== i]) / sqrt(sum(x .== i)),digits=3)
            ax.hist(y[x .== i],x_rang[i],color = MainColors.lcol[i])
            ax.legend([string(my) * "+-" * string(dy)])
            ax.set_xlim([x_rang[i][1],x_rang[i][end]])
        end
        ax.set_title(Legends[i])
    end
    tight_layout()
    savefig(Path_Save * "LHat_" * GorB_labels[i_GorB] * ".pdf")
    savefig(Path_Save * "LHat_" * GorB_labels[i_GorB] * ".png")
    savefig(Path_Save * "LHat_" * GorB_labels[i_GorB] * ".svg")
end
