################################################################################
# Plotting the results of the model recovery analysis for different ranges of l;
# See src/Theory/03_ModelRecovery_data.jl
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

SavePath = "src/Theory/Figures/Recovery/"

# ------------------------------------------------------------------------------
# all-rooms vs. chosen-rooms toggle
# ------------------------------------------------------------------------------
for ifallroom = [true, false]
    if ifallroom      
        nametag = ""
    else
        nametag = "_chosenrooms"
    end

    K = 1
    for β = [10., 2., 1.]
        temp = load(SavePath * "RecSingSub_B" * string(β) * "_K" * string(K)  * nametag *  ".jld2")
        A_agent = temp["A_agent"]
        m_true = temp["m_true"]
        l_true = temp["l_true"]
        m_hat = temp["m_hat"]
        l_hat = temp["l_hat"]
        logl_hat = temp["logl_hat"]


        # ------------------------------------------------------------------------------
        # Confusion matrix
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
        x = log.(l_true[m_true .== 1])
        y = logl_hat[m_true .== 1]
        min_xy, max_xy = -4, 0.5
        ax.plot(x,y,".k",alpha = 0.5)
        ax.plot([min_xy,max_xy],[min_xy,max_xy],"--k")
        pR2 = (1 - mean((y .- x).^2) / mean((y .- mean(x)).^2))
        ax.legend(["pR2 = " * string(round(pR2,digits = 2))])
        if max_xy > 0
            ax.plot([min_xy,max_xy],[0,0],"--k")
            ax.plot([0,0],[min_xy,max_xy],"--k")
        end
        ax.set_xlim([min_xy,max_xy]),ax.set_ylim([min_xy,max_xy])
        ax.set_xticks([min_xy,max_xy]); ax.set_yticks([min_xy,max_xy])
        ax.set_aspect("equal")
        ax.set_xlabel("True log-l"); ax.set_ylabel("Recovered log-l")
        ax.set_title("l-recovery for l < 1; β = " * string(β) * ", K = " * string(K))

        ax = subplot(1,3,3)
        x = log.(l_true[m_true .== 3])
        y = logl_hat[m_true .== 3]
        ax.plot(x,y,".k",alpha = 0.5)
        min_xy, max_xy = -0.5, 2.5
        ax.plot([min_xy,max_xy],[min_xy,max_xy],"--k")
        pR2 = (1 - mean((y .- x).^2) / mean((y .- mean(x)).^2))
        ax.legend(["pR2 = " * string(round(pR2,digits = 2))])
        if min_xy < 0
            ax.plot([min_xy,max_xy],[0,0],"--k")
            ax.plot([0,0],[min_xy,max_xy],"--k")
        end
        ax.set_xlim([min_xy,max_xy]),ax.set_ylim([min_xy,max_xy])
        ax.set_xticks([min_xy,max_xy]); ax.set_yticks([min_xy,max_xy])
        ax.set_aspect("equal")
        ax.set_xlabel("True log-l"); ax.set_ylabel("Recovered log-l")
        ax.set_title("l-recovery for l > 1; β = " * string(β) * ", K = " * string(K))


        tight_layout()

        savefig(SavePath * "Recovery_singlesub_beta_" * string(β) * nametag * 
                        "_K_" * string(K) * ".png")
        savefig(SavePath * "Recovery_singlesub_beta_" * string(β) * nametag * 
                        "_K_" * string(K) * ".pdf")
        savefig(SavePath * "Recovery_singlesub_beta_" * string(β) * nametag * 
                        "_K_" * string(K) * ".svg")
    end
end