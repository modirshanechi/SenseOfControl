################################################################################
# Code for plotting inference results comparing the two blocks
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

Path_Save = "src/GoldExp2Data/Figures/GroupLevelComparative/"
Path_Load_Inf = "src/GoldExp2Data/Figures/"
Path_Load_GInf = "src/GoldExp2Data/Figures/GroupLevel/"
Path_Load = "data/Experiment2/clean/"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
ExcDF = DataFrame(CSV.File(Path_Load * "ExclusionInfo.csv"))
dataDF = DataFrame(CSV.File(Path_Load * "SelectionData.csv"))

subjectIDs = ExcDF.subject[ExcDF.task_outliers .== 0]

# ---------------------------------------------------------------------
# Load group level inference data
# ---------------------------------------------------------------------
BMSdata = load(Path_Load_GInf * "BMS.jld2")
BMSdataLHat = load(Path_Load_GInf * "BMSbasedLHat.jld2")
BMSdataNa  = load(Path_Load_GInf * "BMSbasedNaFit.jld2")

subjectIDsL = [BMSdata["subjectIDsLG"],BMSdata["subjectIDsLB"]]
BMSAllGroups = [BMSdata["BMSAllG"],BMSdata["BMSAllB"]]
BMSEmpGroups = [BMSdata["BMSG"],   BMSdata["BMSB"]]

# ---------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------
Prooms, ΔState, ΔStateDict = gold_proom_sets();
N_rooms = length(Prooms); Ymax = 1; Xmax = 1

GorB = [1,0]
ProcessedDataAll = []
for i_sub = subjectIDs
    @show i_sub
    Path_Load_ID = Path_Load_Inf * "inference_data_sub" * string(i_sub) * ".jld2"
    temp = load(Path_Load_ID)
    chnAll_df = temp["chnAll_df"]
    chnAll_dfG = temp["chnAll_dfG"]; chnemp_dfG = temp["chnemp_dfG"]
    chnAll_dfB = temp["chnAll_dfB"]; chnemp_dfB = temp["chnemp_dfB"]
    
    ProcessedData = []

    pmAll = BMSdata["BMSAll"].exp_M[BMSdata["subjectIDs"] .== i_sub,:][:]
    mAll_hat = findmax(pmAll)[2]

    
    for i_GorB = 1:2
        @show i_GorB
        
        chNa_df = BMSdataNa["chnA_dfsAll"][i_GorB][subjectIDs .== i_sub][1]        
        # ---------------------------------------------------------------------
        # Inferring preferences
        # ---------------------------------------------------------------------
        df = dataDF[dataDF.subject .== i_sub, :]
        df = df[df.timeout .== false, :]
        df = df[df.Gtrials .== GorB[i_GorB],:]
        
        Xinds = [[df.room1[i], df.room2[i]] .+ 1 for i = 1:size(df)[1]]  
        as_inds = df.chosenroom .+ 1
        as = df.action .+ 1
        
        # ---------------------------------------------------------------------
        # model selection
        # ---------------------------------------------------------------------
        pmAll = BMSAllGroups[i_GorB].exp_M[subjectIDs .== i_sub,:][:]
        mAll_hat = findmax(pmAll)[2]

        if i_sub ∈ subjectIDsL[i_GorB]
            pm = BMSEmpGroups[i_GorB].exp_M[subjectIDsL[i_GorB] .== i_sub,:][:]
            m_hat = findmax(pm)[2]
            df_temp = BMSdataLHat["chnemp_dfsAll"][i_GorB][subjectIDsL[i_GorB] .== i_sub][1]
            βs = df_temp.β
            if m_hat == 1
                ls = df_temp.l0
            elseif m_hat == 2
                ls = ones(length(df_temp.l0))
            elseif m_hat == 3
                ls = df_temp.l2
            end
            l_hat = mean(ls)
            dl_hat = std(ls)
            logl_hat = mean(log.(ls))
            dlogl_hat = std(log.(ls))

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
            dl_hat = NaN
            logl_hat = NaN
            dlogl_hat = NaN


            rt = df.rt ./ 1000
            valid_inds = rt .< 10
            rt = rt[valid_inds]
            dv  = abs.([size(Prooms[x[1]])[1] - size(Prooms[x[2]])[1] 
                            for x = Xinds])[valid_inds]
            ρrt = cor(dv,rt)
            rank_ρrt = cor(Rank(dv),Rank(rt))
        end
        
        rt = df.rt ./ 1000
        valid_inds = rt .< 10
        rt = rt[valid_inds]
        dv  = abs.([size(Prooms[x[1]])[1] - size(Prooms[x[2]])[1] 
                            for x = Xinds])[valid_inds]
        ρrt_contNa = cor(dv,rt)
        rank_ρrt_contNa = cor(Rank(dv),Rank(rt))

        
        # ---------------------------------------------------------------------
        # pushin
        # ---------------------------------------------------------------------
        push!(ProcessedData,
            (; pmAll = pmAll, mAll_hat = mAll_hat,
                pm = pm, m_hat = m_hat, 
                l_hat = l_hat, dl_hat = dl_hat, 
                logl_hat=logl_hat, dlogl_hat = dlogl_hat,
                ρrt = ρrt, rank_ρrt = rank_ρrt, 
                ρrt_contNa = ρrt_contNa, rank_ρrt_contNa = rank_ρrt_contNa,
                Cond = df.GB_condition[1]))
    end
    push!(ProcessedDataAll,ProcessedData)
end

# ---------------------------------------------------------------------
# Model selection for the combined trial types
# ---------------------------------------------------------------------
fig = figure(figsize=(12,6)); 

ax = subplot(1,2,1)
x_names = ModelNames.MAllE1
x0 = Array(((0:(length(x_names)-1)) .* 3) .+ 1)
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    # average 
    x = x0 .+ (i_GorB .- 1)
    y = BMSAlltemp.exp_r; dy = BMSAlltemp.d_exp_r; 
    @show y
    ax.bar(x,y, color= MainColors.GB[i_GorB])
    ax.errorbar(x,y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
for i_x = eachindex(x_names)
    p12 = mean(BMSAllGroups[1].R_samples_all[:,i_x] .> 
               BMSAllGroups[2].R_samples_all[:,i_x])
    p12 = round(min(p12, 1 - p12),digits=3)
    ax.text(x0[i_x]+0.5,0.8,string(p12),ha="center")
end
title("Posterior Probabilities for Different Models")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(x_names,fontsize=9)
ax.set_ylabel("E[P(model) | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

ax = subplot(1,2,2)
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    # average 
    x = x0 .+ (i_GorB .- 1)
    y = BMSAlltemp.pxp;
    @show y
    
    ax.bar(x,y, color= MainColors.GB[i_GorB])
    
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
title("Protected exceedence probabilities")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(x_names,fontsize=9)
ax.set_ylabel("P[r_m > r_m' | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

tight_layout()

savefig(Path_Save * "ModelSelAll_expR_combined.pdf")
savefig(Path_Save * "ModelSelAll_expR_combined.png")
savefig(Path_Save * "ModelSelAll_expR_combined.svg")


fig = figure(figsize=(12,6)); 

ax = subplot(1,2,1)
x_names = ModelNames.MEmp
x0 = Array(((0:(length(x_names)-1)) .* 3) .+ 1)
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    # average 
    x = x0 .+ (i_GorB .- 1)
    y = BMSEmptemp.exp_r; dy = BMSEmptemp.d_exp_r; 
    @show y
    for i = eachindex(x)
        ax.bar(x[i],y[i],edgecolor = MainColors.lcol[i], 
                        color= MainColors.GB[i_GorB], linewidth=3)
    end
    ax.errorbar(x,y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
for i_x = eachindex(x_names)
    p12 = mean(BMSAllGroups[1].R_samples_all[:,i_x] .> 
               BMSAllGroups[2].R_samples_all[:,i_x])
    p12 = round(min(p12, 1 - p12),digits=3)
    ax.text(x0[i_x]+0.5,0.8,string(p12),ha="center")
end
title("Posterior Probabilities for Different Models")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(x_names,fontsize=9)
ax.set_ylabel("E[P(model) | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

ax = subplot(1,2,2)
for i_GorB = 1:2
    BMSAlltemp = BMSAllGroups[i_GorB]
    BMSEmptemp = BMSEmpGroups[i_GorB]

    # average 
    x = x0 .+ (i_GorB .- 1)
    y = BMSEmptemp.pxp;
    @show y
    for i = eachindex(x)
        ax.bar(x[i],y[i],edgecolor = MainColors.lcol[i], 
                        color= MainColors.GB[i_GorB], linewidth=3)
    end
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
title("Protected exceedence probabilities")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(x_names,fontsize=9)
ax.set_ylabel("P[r_m > r_m' | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

tight_layout()

savefig(Path_Save * "ModelSel_expR_combined.pdf")
savefig(Path_Save * "ModelSel_expR_combined.png")
savefig(Path_Save * "ModelSel_expR_combined.svg")


# ---------------------------------------------------------------------
# Focuse on double-Emp participants
# ---------------------------------------------------------------------
EmplProcessedDataAll = ProcessedDataAll[(BMSdata["mAll_hatG"] .== 3) .& 
                                        (BMSdata["mAll_hatB"] .== 3)]
lGs   = [d[1].l_hat for d = EmplProcessedDataAll]
dlGs  = [d[1].dl_hat for d = EmplProcessedDataAll]
llGs  = [d[1].logl_hat for d = EmplProcessedDataAll]
dllGs = [d[1].dlogl_hat for d = EmplProcessedDataAll]
lBs   = [d[2].l_hat for d = EmplProcessedDataAll]
llBs  = [d[2].logl_hat for d = EmplProcessedDataAll]
dlBs  = [d[2].dl_hat for d = EmplProcessedDataAll]
dllBs = [d[2].dlogl_hat for d = EmplProcessedDataAll]
Legends = ["G-trials","B-trials"]
Color_plot = MainColors.GB
figure(figsize=(12,8))
ax = subplot(2,3,1)
Ys = [lGs, lBs]
Test_result = OneSampleTTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_OneSampleTTest(Ys...)
for j = eachindex(Ys)
    y_temp = Ys[j]
    my = mean(y_temp); dy = std(y_temp) ./ sqrt(length(y_temp))
    ax.bar(j,my, color = Color_plot[j])
    ax.errorbar(j,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
end
for j = eachindex(Ys[1])
    δ = 0.2 * (rand() - 0.5)
    y_temp = [y[j] for y = Ys]
    ax.plot((1:length(y_temp)) .+ δ, y_temp, ".k", alpha = 0.3)
    ax.plot((1:length(y_temp)) .+ δ, y_temp, "-k", alpha = 0.1)
end
ax.set_ylabel("l-hat")
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))
ax.set_xlim([0,3]); 
ax.set_xticks(1:2); ax.set_xticklabels(Legends)

ax = subplot(2,3,2)
Ys = [lBs, lGs]
ρ = round(cor(Ys...),digits = 2)
ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
ax.plot([x_min,x_max],[x_min,x_max],"--k")
ax.plot([x_min,x_max],[1,1],"--k")
ax.plot([1,1],[x_min,x_max],"--k")
ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
Test_result = CorrelationTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_CorrelationTest(Ys...)
ax.set_ylabel("l-hat-G")
ax.set_xlabel("l-hat-B")
ax.legend(["ρ = " * string(ρ)])
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))

ax = subplot(2,3,3)
Ys = [lBs, lGs]; dYs = [dlBs, dlGs]
ρ = round(cor(Ys...),digits = 2)
ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
ax.errorbar(Ys[1],Ys[2],xerr = dYs[1], yerr=dYs[1],color="k",alpha=0.2,
                linewidth=1,drawstyle="steps",linestyle="",capsize=0)
x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
ax.plot([x_min,x_max],[x_min,x_max],"--k")
ax.plot([x_min,x_max],[1,1],"--k")
ax.plot([1,1],[x_min,x_max],"--k")
ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
Test_result = CorrelationTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_CorrelationTest(Ys...)
ax.set_ylabel("l-hat-G")
ax.set_xlabel("l-hat-B")
ax.legend(["ρ = " * string(ρ)])
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))


ax = subplot(2,3,1+3)
Ys = [llGs, llBs]
Test_result = OneSampleTTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_OneSampleTTest(Ys...)
for j = eachindex(Ys)
    y_temp = Ys[j]
    my = mean(y_temp); dy = std(y_temp) ./ sqrt(length(y_temp))
    ax.bar(j,my, color = Color_plot[j])
    ax.errorbar(j,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
end
for j = eachindex(Ys[1])
    δ = 0.2 * (rand() - 0.5)
    y_temp = [y[j] for y = Ys]
    ax.plot((1:length(y_temp)) .+ δ, y_temp, ".k", alpha = 0.3)
    ax.plot((1:length(y_temp)) .+ δ, y_temp, "-k", alpha = 0.1)
end
ax.set_ylabel("log l-hat")
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))
ax.set_xlim([0,3]); 
ax.set_xticks(1:2); ax.set_xticklabels(Legends)

Ys = [llGs, llBs]
ax = subplot(2,3,2+3)
ρ = round(cor(Ys...),digits = 2)
ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
ax.plot([x_min,x_max],[x_min,x_max],"--k")
ax.plot([x_min,x_max],[0,0],"--k")
ax.plot([0,0],[x_min,x_max],"--k")
ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
Test_result = CorrelationTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_CorrelationTest(Ys...)
ax.set_ylabel("log l-hat-G")
ax.set_xlabel("log l-hat-B")
ax.legend(["ρ = " * string(ρ)])
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))

ax = subplot(2,3,3+3)
Ys = [llBs, llGs]; dYs = [dllBs, dllGs]
ρ = round(cor(Ys...),digits = 2)
ax.plot(Ys[1],Ys[2],".k",alpha = 0.5)
ax.errorbar(Ys[1],Ys[2],xerr = dYs[1], yerr=dYs[1],color="k",alpha=0.2,
                linewidth=1,drawstyle="steps",linestyle="",capsize=0)
x_min = min(ax.get_xlim()[1],ax.get_ylim()[1])
x_max = max(ax.get_xlim()[2],ax.get_ylim()[2])
ax.plot([x_min,x_max],[x_min,x_max],"--k")
ax.plot([x_min,x_max],[0,0],"--k")
ax.plot([0,0],[x_min,x_max],"--k")
ax.set_xlim([x_min,x_max]); ax.set_ylim([x_min,x_max]); ax.set_aspect(1.)
Test_result = CorrelationTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_CorrelationTest(Ys...)
ax.set_ylabel("log l-hat-G")
ax.set_xlabel("log l-hat-B")
ax.legend(["ρ = " * string(ρ)])
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))

tight_layout()
savefig(Path_Save * "ModelSel_lhat_paired.pdf")
savefig(Path_Save * "ModelSel_lhat_paired.png")
savefig(Path_Save * "ModelSel_lhat_paired.svg")


# ---------------------------------------------------------------------
# Focuse on Emp participants
# ---------------------------------------------------------------------
EmplProcessedDataG = ProcessedDataAll[(BMSdata["mAll_hatG"] .== 3)]
EmplProcessedDataB = ProcessedDataAll[(BMSdata["mAll_hatB"] .== 3)]
lGs = [d[1].l_hat for d = EmplProcessedDataG]
lBs = [d[2].l_hat for d = EmplProcessedDataB]
llGs = [d[1].logl_hat for d = EmplProcessedDataG]
llBs = [d[2].logl_hat for d = EmplProcessedDataB]


Legends = ["G-trials","B-trials"]
Color_plot = MainColors.GB
figure(figsize=(8,4))
ax = subplot(1,2,1)
Ys = [lGs, lBs]; 
Test_result = UnequalVarianceTTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_UnequalVarianceTTest(Ys...)
for j = eachindex(Ys)
    y_temp = Ys[j]
    my = mean(y_temp); dy = std(y_temp) ./ sqrt(length(y_temp))
    ax.bar(j,my, color = Color_plot[j])
    ax.errorbar(j,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot(j .+ 0.2 .* (rand(length(y_temp)) .- 0.5), y_temp, 
                            ".k",alpha = 0.5)
end

ax.set_ylabel("l-hat")
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))
ax.set_xlim([0,3]); 
ax.set_xticks(1:2); ax.set_xticklabels(Legends)


ax = subplot(1,2,2)
Ys = [llGs, llBs]; 
Test_result = UnequalVarianceTTest(Ys...)
@show Test_result
pval = pvalue(Test_result)
logBF = BIC_UnequalVarianceTTest(Ys...)
for j = eachindex(Ys)
    y_temp = Ys[j]
    my = mean(y_temp); dy = std(y_temp) ./ sqrt(length(y_temp))
    ax.bar(j,my, color = Color_plot[j])
    ax.errorbar(j,my,yerr=dy,color="k",
                linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot(j .+ 0.2 .* (rand(length(y_temp)) .- 0.5), y_temp, 
                            ".k",alpha = 0.5)
end

ax.set_ylabel("log l-hat")
ax.set_title("pval = " * Func_pval_string(pval) *
            ", lBF = " * Func_logBF_string(logBF))
ax.set_xlim([0,3]); 
ax.set_xticks(1:2); ax.set_xticklabels(Legends)

tight_layout()
savefig(Path_Save * "ModelSel_lhat_separated.pdf")
savefig(Path_Save * "ModelSel_lhat_separated.png")
savefig(Path_Save * "ModelSel_lhat_separated.svg")

# ---------------------------------------------------------------------
# Confusion structure
# ---------------------------------------------------------------------
EmplProcessedDataEmpNa = ProcessedDataAll[
        ((BMSdata["mAll_hatG"] .== 2) .| (BMSdata["mAll_hatG"] .== 3)) .&
        ((BMSdata["mAll_hatB"] .== 2) .| (BMSdata["mAll_hatB"] .== 3))]

mGs = [d[1].m_hat for d = EmplProcessedDataEmpNa]; mGs[isnan.(mGs)] .= 0
mBs = [d[2].m_hat for d = EmplProcessedDataEmpNa]; mBs[isnan.(mBs)] .= 0
mGs .+= 1; mBs .+= 1; 
ConfMatCounts = zeros(4,4)
for i = 1:4
    for j = 1:4
        ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
    end
end
ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

Legends = vcat("Na", ModelNames.MEmp)
fig = figure(figsize=(15,5))
Y = ConfMatRatio
ax = subplot(1,3,1)
cp = ax.imshow(Y, vmax = 0.25, cmap="Purples")
for i = 1:4
    for j = 1:4
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
    end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Overall")

Y = ConfMatCounts
for i = 1:4
    Y[i,:] = Y[i,:] ./ sum(Y[i,:])
end
ax = subplot(1,3,2)
cp = ax.imshow(Y, vmax = 0.7, cmap="Purples")
for i = 1:4
    for j = 1:4
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
    end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Normalized to G")

Y = ConfMatCounts
for i = 1:4
    Y[:,i] = Y[:,i] ./ sum(Y[:,i])
end
ax = subplot(1,3,3)
cp = ax.imshow(Y, vmax = 0.7, cmap="Purples")
for i = 1:4
    for j = 1:4
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
    end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Normalized to B")
tight_layout()

savefig(Path_Save * "ModelSel_ConfMatrix.pdf")
savefig(Path_Save * "ModelSel_ConfMatrix.png")
savefig(Path_Save * "ModelSel_ConfMatrix.svg")

# ---------------------------------------------------------------------
# Confusion structure for separate conditions
# ---------------------------------------------------------------------
GB_legends = ["GB_Cond","BG_Cond"]; GB_conds = [1,0]
for i_GB_cond = 1:2
    mGs = [d[1].m_hat for d = EmplProcessedDataEmpNa]; mGs[isnan.(mGs)] .= 0
    mBs = [d[2].m_hat for d = EmplProcessedDataEmpNa]; mBs[isnan.(mBs)] .= 0
    mGs .+= 1; mBs .+= 1; 

    mGs = mGs[[d[1].Cond for d = EmplProcessedDataEmpNa] .== GB_conds[i_GB_cond]]
    mBs = mBs[[d[1].Cond for d = EmplProcessedDataEmpNa] .== GB_conds[i_GB_cond]]
    ConfMatCounts = zeros(4,4)
    for i = 1:4
        for j = 1:4
            ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
        end
    end
    ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

    Legends = vcat("Na", ModelNames.MEmp)
    fig = figure(figsize=(15,5))
    Y = ConfMatRatio
    ax = subplot(1,3,1)
    cp = ax.imshow(Y,vmax = 0.25, cmap="Purples")
    for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:3); ax.set_xticklabels(Legends)
    ax.set_yticks(0:3); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title("Overall; " * GB_legends[i_GB_cond])

    Y = ConfMatCounts
    for i = 1:4
        Y[i,:] = Y[i,:] ./ sum(Y[i,:])
    end
    ax = subplot(1,3,2)
    cp = ax.imshow(Y,vmax = 0.7, cmap="Purples")
    for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:3); ax.set_xticklabels(Legends)
    ax.set_yticks(0:3); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title("Normalized to G; " * GB_legends[i_GB_cond])

    Y = ConfMatCounts
    for i = 1:4
        Y[:,i] = Y[:,i] ./ sum(Y[:,i])
    end
    ax = subplot(1,3,3)
    cp = ax.imshow(Y,vmax = 0.7, cmap="Purples")
    for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:3); ax.set_xticklabels(Legends)
    ax.set_yticks(0:3); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title("Normalized to B; " * GB_legends[i_GB_cond])
    tight_layout()

    tight_layout()
    savefig(Path_Save * "ModelSel_ConfMatrix_" * GB_legends[i_GB_cond] * ".pdf")
    savefig(Path_Save * "ModelSel_ConfMatrix_" * GB_legends[i_GB_cond] * ".png")
    savefig(Path_Save * "ModelSel_ConfMatrix_" * GB_legends[i_GB_cond] * ".svg")
end

# ---------------------------------------------------------------------
# Confusion structure 2
# ---------------------------------------------------------------------
mGs = [d[1].mAll_hat for d = ProcessedDataAll]
mBs = [d[2].mAll_hat for d = ProcessedDataAll]
ConfMatCounts = zeros(4,4)
for i = 1:4
    for j = 1:4
        ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
    end
end
ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

Legends = ModelNames.MAllE1
fig = figure(figsize=(15,5))
Y = ConfMatRatio
ax = subplot(1,3,1)
cp = ax.imshow(Y, vmax = 0.50, cmap="Purples")
for i = 1:4
    for j = 1:4
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
    end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:3); ax.set_xticklabels(Legends)
ax.set_yticks(0:3); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Overall")

GB_legends = ["GB_Cond","BG_Cond"]; GB_conds = [1,0]
for i_GB_cond = 1:2
    mGs = [d[1].mAll_hat for d = ProcessedDataAll]
    mBs = [d[2].mAll_hat for d = ProcessedDataAll]

    mGs = mGs[[d[1].Cond for d = ProcessedDataAll] .== GB_conds[i_GB_cond]]
    mBs = mBs[[d[1].Cond for d = ProcessedDataAll] .== GB_conds[i_GB_cond]]
    
    ConfMatCounts = zeros(4,4)
    for i = 1:4
        for j = 1:4
            ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
        end
    end
    ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

    ax = subplot(1,3,1 + i_GB_cond)
    Y = ConfMatRatio
    cp = ax.imshow(Y, vmax = 0.50, cmap="Purples")
    for i = 1:4
        for j = 1:4
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:3); ax.set_xticklabels(Legends)
    ax.set_yticks(0:3); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title(GB_legends[i_GB_cond])
end
tight_layout()

savefig(Path_Save * "ModelSelAll_ConfMatrix.pdf")
savefig(Path_Save * "ModelSelAll_ConfMatrix.png")
savefig(Path_Save * "ModelSelAll_ConfMatrix.svg")


# ---------------------------------------------------------------------
# Confusion structure 3
# ---------------------------------------------------------------------
mGs = [d[1].m_hat for d = EmplProcessedDataAll]
mBs = [d[2].m_hat for d = EmplProcessedDataAll]
ConfMatCounts = zeros(3,3)
for i = 1:3
    for j = 1:3
        ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
    end
end
ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

Legends = ModelNames.MEmp
fig = figure(figsize=(15,5))
Y = ConfMatRatio
ax = subplot(1,3,1)
cp = ax.imshow(Y, vmax = 0.50, cmap="Purples")
for i = 1:3
    for j = 1:3
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                        horizontalalignment="center")
    end
end
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:2); ax.set_xticklabels(Legends)
ax.set_yticks(0:2); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Overall")

GB_legends = ["GB_Cond","BG_Cond"]; GB_conds = [1,0]
for i_GB_cond = 1:2
    mGs = [d[1].m_hat for d = EmplProcessedDataAll]
    mBs = [d[2].m_hat for d = EmplProcessedDataAll]

    mGs = mGs[[d[1].Cond for d = EmplProcessedDataAll] .== GB_conds[i_GB_cond]]
    mBs = mBs[[d[1].Cond for d = EmplProcessedDataAll] .== GB_conds[i_GB_cond]]
    
    ConfMatCounts = zeros(3,3)
    for i = 1:3
        for j = 1:3
            ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
        end
    end
    ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

    ax = subplot(1,3,1 + i_GB_cond)
    Y = ConfMatRatio
    cp = ax.imshow(Y, vmax = 0.50, cmap="Purples")
    for i = 1:3
        for j = 1:3
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:2); ax.set_xticklabels(Legends)
    ax.set_yticks(0:2); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title(GB_legends[i_GB_cond])
end
tight_layout()

savefig(Path_Save * "ModelSelOnlyL_ConfMatrix.pdf")
savefig(Path_Save * "ModelSelOnlyL_ConfMatrix.png")
savefig(Path_Save * "ModelSelOnlyL_ConfMatrix.svg")

