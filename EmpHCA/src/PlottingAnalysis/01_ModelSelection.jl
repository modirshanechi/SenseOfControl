################################################################################
# Code for plotting model selection results
# ---> Fig 2E and G1
# ---> Fig 3D and G
# ---> Fig 5C and E1
# ---> Fig S4, S5A, and S10
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using Random, Statistics
using DataFrames
using CSV
using JLD2

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

rng = Xoshiro(2024)

load_name_tag = ""
save_name_tag = "ModelSelection/"

# to whether show the MCMC diagnosis for the BMS results
if_rundiagnosis = true

# ----------------------------------------------------------------------
# Random effect plotting for all models
# ----------------------------------------------------------------------
function plot_BMSAll(BMS, subjectIDs, SavePath; 
                    ModelNamesPlot = ModelNames.MAll)
    Y = BMS.exp_M; Y_names = ["S" * string(s) for s = subjectIDs]
    y = mean(Y, dims=1)[:]; i_winner = findmax(y)[2]
    
    # Heatmap
    sort_inds = sortperm(Y[:,i_winner])
    y = Y[sort_inds,:]; y_names = Y_names[sort_inds]
    fig = figure(figsize = (3,12)); ax = subplot(1,1,1)
    cp = ax.imshow(y, cmap="binary_r",vmin=0.,vmax=1.,aspect="auto")
    ax.set_xticks(0:(size(y)[2]-1)); 
    ax.set_xticklabels(ModelNamesPlot,fontsize=9,rotation = 90)
    ax.set_yticks(0:(size(y)[1]-1)); 
    ax.set_yticklabels(y_names,fontsize=9)
    ax.set_title("all subjects")
    fig.colorbar(cp, ax=ax)
    tight_layout()
    savefig(SavePath * "AllModelSelexpM.pdf")
    savefig(SavePath * "AllModelSelexpM.png")
    savefig(SavePath * "AllModelSelexpM.svg")

    # average
    y = BMS.exp_r; dy = BMS.d_exp_r; x = 1:length(y)
    fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
    ax.bar(x,y, color="k",alpha=0.7)
    ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Posterior Probabilities for Different Models")
    ax.set_xticks(x)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("E[P(model) | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    y = BMS.pxp; x = 1:length(y)
    ax = subplot(1,2,2)
    ax.bar(x,y, color="k")
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Protected exceedence probabilities")
    ax.set_xticks(x)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("P[r_m > r_m' | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    tight_layout()
    savefig(SavePath * "AllModelSelexpR.pdf")
    savefig(SavePath * "AllModelSelexpR.png")
    savefig(SavePath * "AllModelSelexpR.svg")
end

# ----------------------------------------------------------------------
# Random effect plotting for l-range models
# ----------------------------------------------------------------------
function plot_BMSEmp(BMS, subjectIDs, SavePath; 
                    ModelNamesPlot = ModelNames.MEmp)
    Y = BMS.exp_M; Y_names = ["S" * string(s) for s = subjectIDs]
    y = mean(Y, dims=1)[:]; i_winner = findmax(y)[2]
    
    # Heatmap
    sort_inds = sortperm(Y[:,i_winner])
    y = Y[sort_inds,:]; y_names = Y_names[sort_inds]
    fig = figure(figsize = (3,12)); ax = subplot(1,1,1)
    cp = ax.imshow(y, cmap="binary_r",vmin=0.,vmax=1.,aspect="auto")
    ax.set_xticks(0:(size(y)[2]-1)); 
    ax.set_xticklabels(ModelNamesPlot,fontsize=9,rotation = 90)
    ax.set_yticks(0:(size(y)[1]-1)); 
    ax.set_yticklabels(y_names,fontsize=9)
    ax.set_title("emp-l")
    fig.colorbar(cp, ax=ax)
    tight_layout()
    savefig(SavePath * "EmpModelSelexpM.pdf")
    savefig(SavePath * "EmpModelSelexpM.png")
    savefig(SavePath * "EmpModelSelexpM.svg")


    figure(figsize=(5,5))
    ax = subplot(1,1,1)
    Y = [BMS.exp_M[i,:] for i = eachindex(subjectIDs)]
    y_col = [findmax(y)[2] for y = Y]
    ax.plot([0,1],[0,0],"k",alpha = 0.2)
    ax.plot([0,0],[0,1],"k",alpha = 0.2)
    ax.plot([0,1],[1,0],"k",alpha = 0.2)
    ax.set_xlim([-0.05,1.05]); ax.set_ylim([-0.05,1.05]); 
    ax.set_aspect("equal", "box")
    for i = eachindex(Y)
        y = Y[i] .+ 0.03 .* (rand(rng) - 0.5)
        ax.plot(y[1],y[3],".",color= MainColors.lcol[y_col[i]],alpha=0.5)
    end
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_ylabel("P(l > 1| data)"); ax.set_xlabel("P(l < 1| data)")
    tight_layout()
    savefig(SavePath * "EmpModelSelexpMtri.pdf")
    savefig(SavePath * "EmpModelSelexpMtri.png")
    savefig(SavePath * "EmpModelSelexpMtri.svg")

    # average
    y = BMS.exp_r; dy = BMS.d_exp_r; x = 1:length(y)
    fig = figure(figsize=(12,6)); ax = subplot(1,2,1)
    for i = eachindex(x)
        ax.bar(x[i],y[i], color=MainColors.lcol[i])
    end
    ax.errorbar(1:length(y),y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Posterior Probabilities for Different Models")
    ax.set_xticks(x)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("E[P(model) | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    y = BMS.pxp; x = 1:length(y)
    ax = subplot(1,2,2)
    for i = eachindex(x)
        ax.bar(x[i],y[i], color=MainColors.lcol[i])
    end
    ax.plot([x[1]-1,x[end]+1],[1,1] ./ length(y), 
                linestyle="dashed",linewidth=1,color="k")
    title("Protected exceedence probabilities")
    ax.set_xticks(x)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("P[r_m > r_m' | Data ]")
    ax.set_xlim([x[1]-1,x[end]+1])
    ax.set_ylim([0,1.0])

    tight_layout()
    savefig(SavePath * "EmpModelSelexpR.pdf")
    savefig(SavePath * "EmpModelSelexpR.png")
    savefig(SavePath * "EmpModelSelexpR.svg")
end

function print_BMSdiag(BMS)
    @show BMS.r_diagnostics
    println("Fraction of converged M-indicators:")
    @show mean(BMS.M_diagnostics.pass)
    if mean(BMS.M_diagnostics.pass) < 1.0
        println("Not-converged:")
        @show BMS.M_diagnostics[BMS.M_diagnostics.pass .== 0, : ]
    end
    println("Fraction of constant M-indicators:")
    @show mean(BMS.M_diagnostics.constant)
    println("BOR relative error:")
    @show mean(BMS.d_BOR / BMS.BOR)
end

# ----------------------------------------------------------------------
# Loop over the three experiments
# ----------------------------------------------------------------------
for i_exp = 1:3
    # ----------------------------------------------------------------------
    # Loading data
    # ----------------------------------------------------------------------
    expspec = ExperimentSpecification(i_exp)
    data = load_clean_data(expspec);
    subjectIDs = data.subjectIDs

    # ----------------------------------------------------------------------
    # Loading inference data
    # ----------------------------------------------------------------------
    BMSdata = load(expspec.fig_path * load_name_tag * "BMS.jld2")

    # ----------------------------------------------------------------------
    # Plotting BMS
    # ----------------------------------------------------------------------
    if !expspec.has_gb_split
        plot_BMSAll(BMSdata["BMSAll"], subjectIDs, 
                    expspec.fig_path * save_name_tag)
        
        plot_BMSEmp(BMSdata["BMSEmp"], BMSdata["EmpsubjectIDs"], 
                expspec.fig_path * save_name_tag)
        if if_rundiagnosis
            println("------------------------------------")
            @show i_exp
            println("MCMC diagnosis for BMS all:")
            println("------------------------------------")
            print_BMSdiag(BMSdata["BMSAll"])

            println("------------------------------------")
            println("MCMC diagnosis for BMS Emp:")
            println("------------------------------------")
            print_BMSdiag(BMSdata["BMSEmp"])
        end

    else
        plot_BMSAll(BMSdata["BMSAll"][1], subjectIDs, 
                    expspec.fig_path * save_name_tag * "Gold")
        plot_BMSAll(BMSdata["BMSAll"][2], subjectIDs, 
                    expspec.fig_path * save_name_tag * "Bomb")
        if if_rundiagnosis
            println("------------------------------------")
            @show i_exp
            println("MCMC diagnosis for BMS all; gold:")
            println("------------------------------------")
            print_BMSdiag(BMSdata["BMSAll"][1])
            println("------------------------------------")
            println("MCMC diagnosis for BMS all; bomb:")
            println("------------------------------------")
            print_BMSdiag(BMSdata["BMSAll"][2])
        end
    end
end


# ----------------------------------------------------------------------
# Only for experiment 2
# ----------------------------------------------------------------------
i_exp = 2

# ----------------------------------------------------------------------
# Loading data
# ----------------------------------------------------------------------
expspec = ExperimentSpecification(i_exp)
data = load_clean_data(expspec);
subjectIDs = data.subjectIDs

# ----------------------------------------------------------------------
# Loading inference data
# ----------------------------------------------------------------------
BMSdata = load(expspec.fig_path * load_name_tag * "BMS.jld2")
BMSAll = BMSdata["BMSAll"];

BMSEmpSet = [BMSdata["BMSEmp"],BMSdata["BMSEmpalone"], BMSdata["BMSEmpGen"]]
BMSEmpNames = ["Gen+Emp", "EmpAlone", "General"]

if if_rundiagnosis
    for i = eachindex(BMSEmpSet)
        BMStemp = BMSEmpSet[i]
        println("------------------------------------")
        @show i_exp
        @show BMSEmpNames[i]
        println("MCMC diagnosis for BMS all; gold:")
        println("------------------------------------")
        print_BMSdiag(BMStemp[1])
        println("------------------------------------")
        println("MCMC diagnosis for BMS all; bomb:")
        println("------------------------------------")
        print_BMSdiag(BMStemp[2])
    end
end

# ---------------------------------------------------------------------
# Model selection for the combined trial types
# ---------------------------------------------------------------------
fig = figure(figsize=(12,6)); 

ax = subplot(1,2,1)
ModelNamesPlot = ModelNames.MAll
x0 = Array(((0:(length(ModelNamesPlot)-1)) .* 3) .+ 1)
for i_GorB = 1:2
    BMSAlltemp = BMSAll[i_GorB]

    # average
    x = x0 .+ (i_GorB .- 1)
    y = BMSAlltemp.exp_r; dy = BMSAlltemp.d_exp_r;
    ax.bar(x,y, color= MainColors.GB[i_GorB])
    ax.errorbar(x,y[:],yerr=dy[:],color="k",
                    linewidth=1,drawstyle="steps",linestyle="",capsize=3)
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
for i_x = eachindex(ModelNamesPlot)
    p12 = mean(BMSAll[1].R_samples_all[:,i_x] .> 
               BMSAll[2].R_samples_all[:,i_x])
    p12 = round(min(p12, 1 - p12),digits=3)
    ax.text(x0[i_x]+0.5,0.8,string(p12),ha="center")
end
title("Posterior Probabilities for Different Models")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(ModelNamesPlot,fontsize=9)
ax.set_ylabel("E[P(model) | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

ax = subplot(1,2,2)
for i_GorB = 1:2
    BMSAlltemp = BMSAll[i_GorB]

    # average
    x = x0 .+ (i_GorB .- 1)
    y = BMSAlltemp.pxp;
    
    ax.bar(x,y, color= MainColors.GB[i_GorB])
    
end
ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
            linestyle="dashed",linewidth=1,color="k")
title("Protected exceedence probabilities")
ax.set_xticks(x0 .+ 0.5)
ax.set_xticklabels(ModelNamesPlot,fontsize=9)
ax.set_ylabel("P[r_m > r_m' | Data ]")
ax.set_xlim([x0[1]-1,x0[end]+2])
ax.set_ylim([0,1.0])

tight_layout()

savefig(expspec.fig_path * save_name_tag * "ModelSelAll_expR_combined.pdf")
savefig(expspec.fig_path * save_name_tag * "ModelSelAll_expR_combined.png")
savefig(expspec.fig_path * save_name_tag * "ModelSelAll_expR_combined.svg")


for i_plot = eachindex(BMSEmpNames)
    BMSEmp = BMSEmpSet[i_plot]
    fig = figure(figsize=(12,6)); 

    ax = subplot(1,2,1)
    ModelNamesPlot = ModelNames.MEmp
    x0 = Array(((0:(length(ModelNamesPlot)-1)) .* 3) .+ 1)
    for i_GorB = 1:2
        BMSEmptemp = BMSEmp[i_GorB]

        # average
        x = x0 .+ (i_GorB .- 1)
        y = BMSEmptemp.exp_r; dy = BMSEmptemp.d_exp_r;
        for i = eachindex(x)
            ax.bar(x[i],y[i], color= MainColors.GB[i_GorB], linewidth=3)
        end
        ax.errorbar(x,y[:],yerr=dy[:],color="k",
                        linewidth=1,drawstyle="steps",linestyle="",capsize=3)
    end
    ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
                linestyle="dashed",linewidth=1,color="k")
    for i_x = eachindex(ModelNamesPlot)
        p12 = mean(BMSEmp[1].R_samples_all[:,i_x] .> 
                BMSEmp[2].R_samples_all[:,i_x])
        p12 = round(min(p12, 1 - p12),digits=3)
        ax.text(x0[i_x]+0.5,0.8,string(p12),ha="center")
    end
    title("Posterior Probabilities; " * BMSEmpNames[i_plot] * 
            "; n = (" * string(size(BMSEmp[1].exp_M)[1]) * 
            ", " * string(size(BMSEmp[2].exp_M)[1]) * ")")
    ax.set_xticks(x0 .+ 0.5)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("E[P(model) | Data ]")
    ax.set_xlim([x0[1]-1,x0[end]+2])
    ax.set_ylim([0,1.0])

    ax = subplot(1,2,2)
    for i_GorB = 1:2
        BMSEmptemp = BMSEmp[i_GorB]

        # average
        x = x0 .+ (i_GorB .- 1)
        y = BMSEmptemp.pxp;
        for i = eachindex(x)
            ax.bar(x[i],y[i], color= MainColors.GB[i_GorB])
        end
    end
    ax.plot([x0[1]-1,x0[end]+2],[1,1] ./ length(x0), 
                linestyle="dashed",linewidth=1,color="k")
    title("Protected exceedence probabilities")
    ax.set_xticks(x0 .+ 0.5)
    ax.set_xticklabels(ModelNamesPlot,fontsize=9)
    ax.set_ylabel("P[r_m > r_m' | Data ]")
    ax.set_xlim([x0[1]-1,x0[end]+2])
    ax.set_ylim([0,1.0])

    tight_layout()

    savefig(expspec.fig_path * save_name_tag * "ModelSel_expR_combined" * BMSEmpNames[i_plot] * ".pdf")
    savefig(expspec.fig_path * save_name_tag * "ModelSel_expR_combined" * BMSEmpNames[i_plot] * ".png")
    savefig(expspec.fig_path * save_name_tag * "ModelSel_expR_combined" * BMSEmpNames[i_plot] * ".svg")
end

# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------
mGs = [findmax(BMSAll[1].exp_M[i,:])[2] for i = 1:size(BMSAll[1].exp_M)[1]]
mBs = [findmax(BMSAll[2].exp_M[i,:])[2] for i = 1:size(BMSAll[2].exp_M)[1]]
BGs = [data.selDF.GB_condition[data.selDF.subject .== i][1] for i = subjectIDs]

Legends = ModelNames.MAll; N_model = length(Legends)

ConfMatCounts = zeros(N_model,N_model)
for i = 1:N_model
    for j = 1:N_model
        ConfMatCounts[i,j] = sum((mGs .== i) .& (mBs .== j))
    end
end
ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

fig = figure(figsize=(15,5))
Y = ConfMatRatio
ax = subplot(1,3,1)
cp = ax.imshow(Y, vmax = 0.40, cmap="Purples")
for i = 1:N_model
    for j = 1:N_model
            ax.text(j - 1, i - 1, string(round(Y[i,j],digits=3)),
                        horizontalalignment="center")
    end
end
@show sum(round.(Y,digits=3))
fig.colorbar(cp, ax=ax)
ax.set_xticks(0:(N_model-1)); ax.set_xticklabels(Legends)
ax.set_yticks(0:(N_model-1)); ax.set_yticklabels(Legends)
ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
ax.set_title("Overall")

GB_legends = ["GB_Cond","BG_Cond"]; GB_conds = [1,0]
for i_GB_cond = 1:2
    mGs_plot = mGs[BGs .== GB_conds[i_GB_cond]]
    mBs_plot = mBs[BGs .== GB_conds[i_GB_cond]]
    
    ConfMatCounts = zeros(N_model,N_model)
    for i = 1:N_model
        for j = 1:N_model
            ConfMatCounts[i,j] = sum((mGs_plot .== i) .& (mBs_plot .== j))
        end
    end
    ConfMatRatio = ConfMatCounts ./ sum(ConfMatCounts)

    ax = subplot(1,3,1 + i_GB_cond)
    Y = ConfMatRatio
    cp = ax.imshow(Y, vmax = 0.40, cmap="Purples")
    for i = 1:N_model
        for j = 1:N_model
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=3)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:(N_model-1)); ax.set_xticklabels(Legends)
    ax.set_yticks(0:(N_model-1)); ax.set_yticklabels(Legends)
    ax.set_ylabel("G-trials"); ax.set_xlabel("B-trials")
    ax.set_title(GB_legends[i_GB_cond])
end
tight_layout()

savefig(expspec.fig_path * save_name_tag *  "ModelSelAll_ConfMatrix.pdf")
savefig(expspec.fig_path * save_name_tag *  "ModelSelAll_ConfMatrix.png")
savefig(expspec.fig_path * save_name_tag *  "ModelSelAll_ConfMatrix.svg")
