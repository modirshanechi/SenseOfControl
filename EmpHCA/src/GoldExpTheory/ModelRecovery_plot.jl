################################################################################
# Plotting the results of the model recovery analysis for different ranges of l;
# See src/GoldExpTheory/ModelRecovery_data.jl
################################################################################
using PyPlot
using EmpHCA
using LinearAlgebra
using NNlib: softmax
using Random
using Turing, MCMCChains, Distributions
using DataFrames
using JLD2
using AdvancedMH

import StatsPlots

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

Path_Save = "src/GoldExpTheory/Figures/Recovery/"

# ------------------------------------------------------------------------------
# possible rooms
# ------------------------------------------------------------------------------
ifallroom = false
if ifallroom      
      nametag = ""
else
      nametag = "_chosenrooms"
end

for K = [1] 
for β = [10., 2., 1.]
    temp = load(Path_Save * "RecSingSub_B" * string(β) * "_K" * string(K)  * nametag *  ".jld2")
    A_agent = temp["A_agent"]
    chn_set = temp["chn_set"]
    m_true = temp["m_true"]
    l_true = temp["l_true"]
    γ_true = temp["γ_true"]
    m_hat = temp["m_hat"]
    l_hat = temp["l_hat"]
    γ_hat = temp["γ_hat"]
    β_hat = temp["β_hat"]


    # ------------------------------------------------------------------------------
    # Turing inference: param
    # ------------------------------------------------------------------------------
    Legends = ["l < 1", "l = 1", "l > 1"]
    ConfMat = zeros(3,3)
    for i = 1:3
        for j = 1:3
                ConfMat[i,j] = sum(m_hat[m_true .== i] .== j)
        end
        ConfMat[i,:] .= ConfMat[i,:] ./ sum(ConfMat[i,:])
    end


    fig = figure(figsize=(15,5))
    Y = ConfMat
    ax = subplot(1,3,1)
    cp = ax.imshow(Y,vmin=0,vmax=1.0,cmap="Blues")
    for i = 1:3
        for j = 1:3
                ax.text(j - 1, i - 1, string(round(Y[i,j],digits=2)),
                            horizontalalignment="center")
        end
    end
    fig.colorbar(cp, ax=ax)
    ax.set_xticks(0:2); ax.set_xticklabels(Legends)
    ax.set_yticks(0:2); ax.set_yticklabels(Legends)
    ax.set_ylabel("True model"); ax.set_xlabel("Recovered model")
    ax.set_title("Recovery rate with β = " * string(β) * ", K = " * string(K))

    ax = subplot(1,3,2)
    x = l_true[m_true .== 1]
    y = l_hat[m_true .== 1]
    ax.plot(x,y,".k",alpha = 0.5)
    ax.plot([0,1],[0,1],"--k")
    # ax.legend(["ρ = " * string(round(cor(x,y),digits = 2))])
    pR2 = (1 - mean((y .- x).^2) / mean((y .- mean(x)).^2))
    ax.legend(["pR2 = " * string(round(pR2,digits = 2))])
    ax.set_xlim([0,1]),ax.set_ylim([0,1])
    ax.set_aspect("equal")
    ax.set_xlabel("True l"); ax.set_ylabel("Recovered l")
    ax.set_title("l-recovery for l < 1; β = " * string(β) * ", K = " * string(K))

    ax = subplot(1,3,3)
    x = l_true[m_true .== 3]
    y = l_hat[m_true .== 3]
    ax.plot(x,y,".k",alpha = 0.5)
    ax.plot([1,4],[1,4],"--k")
    # ax.legend(["ρ = " * string(round(cor(x,y),digits = 2))])
    pR2 = (1 - mean((y .- x).^2) / mean((y .- mean(x)).^2))
    ax.legend(["pR2 = " * string(round(pR2,digits = 2))])
    ax.set_xlim([1,4]),ax.set_ylim([1,4])
    ax.set_aspect("equal")
    ax.set_xlabel("True l"); ax.set_ylabel("Recovered l")
    ax.set_title("l-recovery for l > 1; β = " * string(β) * ", K = " * string(K))


    tight_layout()

    savefig(Path_Save * "Recovery_singlesub_beta_" * string(β) * nametag * 
                    "_K_" * string(K) * ".png")
    savefig(Path_Save * "Recovery_singlesub_beta_" * string(β) * nametag * 
                    "_K_" * string(K) * ".pdf")
    savefig(Path_Save * "Recovery_singlesub_beta_" * string(β) * nametag * 
                    "_K_" * string(K) * ".svg")
end
end